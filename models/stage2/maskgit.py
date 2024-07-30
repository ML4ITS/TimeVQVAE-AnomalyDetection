import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from pathlib import Path
import tempfile
from typing import Union

from einops import repeat, rearrange
from typing import Callable
from models.stage2.bidirectional_transformer import BidirectionalTransformer
from collections import deque
from einops.layers.torch import Rearrange
from torchtune.modules import RotaryPositionalEmbeddings

from experiments.exp_stage1 import ExpStage1
from models.stage1.vq_vae_encdec import VQVAEEncoder, VQVAEDecoder
from models.stage1.vq import VectorQuantize

from utils import compute_downsample_rate, get_root_dir, freeze, timefreq_to_time, time_to_timefreq, quantize, zero_pad_low_freq, zero_pad_high_freq


def cov_loss_fn(codebook):
    """
    - codebook: (K, dim)
    """
    norm_cb = (codebook - codebook.mean(dim=1, keepdim=True))  # (K d)
    norm_cb = F.normalize(norm_cb, p=2, dim=1)  # (K d)
    corr_mat = torch.mm(norm_cb, norm_cb.T)  # (K K)
    corr_mat.fill_diagonal_(0.)
    cov_loss = (corr_mat ** 2).mean()
    return cov_loss


def var_loss_fn(codebook, min_scale=1.):
    """
    - codebook: (K dim)
    """
    std_cb = torch.sqrt(codebook.var(dim=1) + 1e-4)
    var_loss = torch.mean(F.relu(min_scale - std_cb))
    return var_loss

def gamma_func(mode="cosine"):
    if mode == "linear":
        return lambda r: 1 - r
    elif mode == "cosine":
        return lambda r: np.cos(r * np.pi / 2)
    elif mode == "square":
        return lambda r: 1 - r ** 2
    elif mode == "cubic":
        return lambda r: 1 - r ** 3
    else:
        raise NotImplementedError

class MaskGIT(nn.Module):
    """
    ref: https://github.com/dome272/MaskGIT-pytorch/blob/cff485ad3a14b6ed5f3aa966e045ea2bc8c68ad8/transformer.py#L11
    """

    def __init__(self,
                 dataset_idx:int,
                 input_length: int,
                 choice_temperature: int,
                 T: int,
                 config: dict,
                 *args, **kwargs):
        super().__init__()
        self.choice_temperature = choice_temperature
        self.T = T
        self.config = config
        self.n_fft = self.config['VQ-VAE']['n_fft']
        self.mask_token_id = config['VQ-VAE']['codebook_size']
        self.gamma = gamma_func(config['MaskGIT']['mask_scheduling_func'])
        self.input_length = input_length

        self.stage1 = ExpStage1.load_from_checkpoint(os.path.join('saved_models', f'stage1-{dataset_idx}.ckpt'), 
                                                     input_length=input_length, 
                                                     config=config,
                                                     map_location='cpu')
        freeze(self.stage1)
        self.stage1.eval()
        
        self.encoder = self.stage1.encoder
        self.decoder = self.stage1.decoder
        self.vq_model = self.stage1.vq_model

        self.num_tokens = self.encoder.num_tokens.item()
        self.H_prime, self.W_prime = self.encoder.H_prime, self.encoder.W_prime

        # transformer
        self.transformer = BidirectionalTransformer(embed_dim=config['encoder']['dim'],
                                                    num_tokens=self.num_tokens,
                                                    codebook_size=config['VQ-VAE']['codebook_size'],
                                                    **config['MaskGIT']['prior_model'],
                                                    )

    @torch.no_grad()
    def encode_to_z_q(self, x, encoder: VQVAEEncoder, vq_model: VectorQuantize, svq_temp=None):
        """
        x: (b c l)
        """
        z = encoder(x)
        zq, s, _, _ = quantize(z, vq_model, svq_temp=svq_temp)  # (b c h w), (b (h w) h), ...
        return zq, s

    def forward(self, x):
        """
        x: (B, C, L)
        REF: [https://github.com/dome272/MaskGIT-pytorch/blob/main/transformer.py]
        """
        device = x.device
        self.encoder.eval()
        self.decoder.eval()
        self.vq_model.eval()

        with torch.no_grad():
            zq, s = self.encode_to_z_q(x, self.encoder, self.vq_model)  # (b n)

        # masked prediction
        s_M, mask = self._randomly_mask_tokens(s, self.mask_token_id, device)
        logits = self.transformer(s_M)  # (b n codebook_size)
        
        # loss
        logits_on_mask = logits[~mask]
        s_on_mask = s[~mask]
        mask_pred_loss = F.cross_entropy(logits_on_mask.float(), s_on_mask.long())

        return mask_pred_loss

    def _randomly_mask_tokens(self, s, mask_token_id, device):
        """
        s: token set
        """
        b, n = s.shape
        
        # sample masking indices
        ratio = np.random.uniform(0, 1, (b,))  # (b,)
        n_unmasks = np.floor(self.gamma(ratio) * n)  # (b,)
        n_unmasks = np.clip(n_unmasks, a_min=0, a_max=n-1).astype(int)  # ensures that there's at least one masked token
        rand = torch.rand((b, n), device=device)  # (b n)
        mask = torch.zeros((b, n), dtype=torch.bool, device=device)  # (b n)

        for i in range(b):
            ind = rand[i].topk(n_unmasks[i], dim=-1).indices
            mask[i].scatter_(dim=-1, index=ind, value=True)

        # mask the token set
        masked_indices = mask_token_id * torch.ones((b, n), device=device)  # (b n)
        s_M = mask * s + (~mask) * masked_indices  # (b n); `~` reverses bool-typed data
        return s_M.long(), mask
    
    def _create_input_tokens_normal(self, num, num_tokens, mask_token_id, device):
        """
        returns masked tokens
        """
        blank_tokens = torch.ones((num, num_tokens), device=device)
        masked_tokens = mask_token_id * blank_tokens
        return masked_tokens.to(torch.int64)

    def mask_by_random_topk(self, mask_len, probs, temperature=1.0, device='cpu'):
        """
        mask_len: (b 1)
        probs: (b n); also for the confidence scores

        This version keeps `mask_len` exactly.
        """
        def log(t, eps=1e-20):
            return torch.log(t.clamp(min=eps))

        def gumbel_noise(t):
            """
            Gumbel max trick: https://neptune.ai/blog/gumbel-softmax-loss-function-guide-how-to-implement-it-in-pytorch
            """
            noise = torch.zeros_like(t).uniform_(0, 1)
            return -log(-log(noise))

        confidence = torch.log(probs + 1e-5) + temperature * gumbel_noise(probs).to(device)  # Gumbel max trick; 1e-5 for numerical stability; (b n)
        mask_len_unique = int(mask_len.unique().item())
        masking_ind = torch.topk(confidence, k=mask_len_unique, dim=-1, largest=False).indices  # (b k)
        masking = torch.zeros_like(confidence).to(device)  # (b n)
        for i in range(masking_ind.shape[0]):
            masking[i, masking_ind[i].long()] = 1.
        masking = masking.bool()
        return masking

    def _parallel_unmasking(self,
                            s: torch.LongTensor,
                            unknown_number_in_the_beginning,
                            device,
                            init_t:int=0,
                            ):
        for t in range(init_t, self.T):
            logits = self.transformer(s)  # (b n codebook_size) == (b n K)
            sampled_ids = torch.distributions.categorical.Categorical(logits=logits).sample()  # (b n)
            unknown_map = (s == self.mask_token_id)  # which tokens need to be sampled; (b n)
            sampled_ids = torch.where(unknown_map, sampled_ids, s)  # keep the previously-sampled tokens; (b n)

            # create masking according to `t`
            ratio = 1. * (t + 1) / self.T  # just a percentage e.g. 1 / 12
            mask_ratio = self.gamma(ratio)

            probs = F.softmax(logits, dim=-1)  # convert logits into probs; (b n K)
            selected_probs = torch.gather(probs, dim=-1, index=sampled_ids[:, :, None])[:, :, 0]  # get probability for the selected tokens; p(\hat{s}(t) | \hat{s}_M(t)); (b n)
            _CONFIDENCE_OF_KNOWN_TOKENS = torch.Tensor([torch.inf]).to(device)
            selected_probs = torch.where(unknown_map, selected_probs, _CONFIDENCE_OF_KNOWN_TOKENS)  # assign inf probability to the previously-selected tokens; (b n)

            mask_len = torch.unsqueeze(torch.floor(unknown_number_in_the_beginning * mask_ratio), 1)  # number of tokens that are to be masked;  (b,)
            mask_len = torch.clip(mask_len, min=0.)  # `mask_len` should be equal or larger than zero.

            # Adds noise for randomness
            masking = self.mask_by_random_topk(mask_len, selected_probs, temperature=self.choice_temperature * (1. - ratio), device=device)

            # Masks tokens with lower confidence.
            s = torch.where(masking, self.mask_token_id, sampled_ids)  # (b n)

        return s

    @torch.no_grad()
    def iterative_decoding(self, num=1, device='cpu'):
        """
        It performs the iterative decoding and samples token indices.
        :param num: number of samples
        :return: sampled token indices
        """
        s = self._create_input_tokens_normal(num, self.num_tokens, self.mask_token_id, device)  # (b n)

        unknown_number_in_the_beginning = torch.sum(s == self.mask_token_id, dim=-1)  # (b,)

        s = self._parallel_unmasking(s, unknown_number_in_the_beginning, device)
        return s

    @torch.no_grad()
    def explainable_sampling(self, t_star:int, s_star:torch.LongTensor):
        """
        t_star: (int)
        s_star: (n_samples, n) = (b n)
        """
        b, n = s_star.shape
        device = s_star.device

        unknown_number_in_the_beginning = torch.ones((b,), device=device) * self.mask_token_id  # (b,)
        s = self._parallel_unmasking(s_star, unknown_number_in_the_beginning, device, init_t=t_star)  # (b n)
        return s

    def decode_token_ind_to_timeseries(self, s: torch.Tensor, return_representations: bool = False):
        """
        It takes token embedding indices and decodes them to time series.
        :param s: token embedding index
        :param return_representations:
        :return:
        """
        zq = F.embedding(s, self.vq_model._codebook.embed)  # (b n d)
        zq = self.vq_model.project_out(zq)  # (b n c)
        zq = rearrange(zq, 'b n c -> b c n')  # (b c n) == (b c (h w))
        zq = rearrange(zq, 'b c (h w) -> b c h w', h=self.H_prime, w=self.W_prime)

        xhat = self.decoder(zq)  # (b c l)

        if return_representations:
            return xhat, zq
        else:
            return xhat
