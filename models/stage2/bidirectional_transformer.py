import math
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from x_transformers import ContinuousTransformerWrapper, Encoder as TFEncoder


def load_pretrained_tok_emb(pretrained_tok_emb, tok_emb):
    """
    :param pretrained_tok_emb: pretrained token embedding from stage 1
    :param tok_emb: token embedding of the transformer
    :return:
    """
    with torch.no_grad():
        if pretrained_tok_emb != None:
            tok_emb.weight[:-1, :] = pretrained_tok_emb


class BidirectionalTransformer(nn.Module):
    def __init__(self,
                 num_tokens: int,
                 codebook_size: int,
                 embed_dim: int,
                 hidden_dim: int,
                 depth: int,
                 heads: int,
                 attn_dim_head: int,
                 ff_mult: int,
                 use_rmsnorm: bool,
                 dropout:float,
                 **kwargs):
        """
        :param kind:
        :param num_tokens:
        :param codebook_sizes:
        :param embed_dim:
        :param hidden_dim:
        :param depth:
        :param heads:
        :param ff_mult:
        :param use_rmsnorm:
        :param pretrained_tok_emb_l: if given, the embedding of the transformer is initialized with the pretrained embedding from stage 1; low-frequency
        :param pretrained_tok_emb_h: if given, the embedding of the transformer is initialized with the pretrained embedding from stage 1; high-frequency
        :param freeze_pretrained_tokens:
        :param num_tokens_l:
        :param kwargs:
        """
        super().__init__()
        self.num_tokens = num_tokens
        in_dim = embed_dim
        out_dim = embed_dim
        self.dropout = dropout
        self.mask_token_idx = codebook_size

        # token embeddings
        self.tok_emb = nn.Embedding(codebook_size+1, embed_dim)  # `+1` is for mask-token

        # transformer
        self.pos_emb = nn.Embedding(self.num_tokens + 1, in_dim)
        self.blocks = ContinuousTransformerWrapper(dim_in=in_dim,
                                                   dim_out=in_dim,
                                                   max_seq_len=self.num_tokens + 1,
                                                   use_abs_pos_emb=False,
                                                   post_emb_norm=True,  # seems to be quite helpful
                                                   attn_layers=TFEncoder(
                                                       pre_norm=True,
                                                       dim=hidden_dim,
                                                       depth=depth,
                                                       heads=heads,
                                                       attn_dim_head=attn_dim_head,
                                                       use_rmsnorm=use_rmsnorm,
                                                       ff_mult=ff_mult,
                                                       layer_dropout=dropout,   # stochastic depth - dropout entire layer
                                                       attn_dropout=dropout,    # dropout post-attention
                                                       ff_dropout=dropout       # feedforward dropout
                                                   ))
        self.pred_head = nn.Sequential(*[
            nn.Linear(in_features=in_dim, out_features=out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim, eps=1e-12)
        ])
        self.bias = nn.Parameter(torch.zeros(self.num_tokens, codebook_size+1))

    def forward(self, s_M:torch.LongTensor):
        """
        s_M: (b n)
        """
        b, n = s_M.shape

        token_embeddings = self.tok_emb(s_M)  # (b n dim)
        if self.training:
            mask_ind = (s_M == self.mask_token_idx)[:,:,None]  # (b n 1)
            token_embeddings_dropout = F.dropout(token_embeddings, p=self.dropout)  # (b n d)
            token_embeddings = torch.where(mask_ind, token_embeddings, token_embeddings_dropout)  # (b n d)

        position_embeddings = self.pos_emb.weight[:n, :]
        embed = token_embeddings + position_embeddings  # (b, n, dim)

        embed = self.blocks(embed)  # (b, n, dim)
        embed = self.pred_head(embed)  # (b n d)

        logits = torch.matmul(embed, self.tok_emb.weight.T) + self.bias  # (b, n, codebook_size+1)
        logits = logits[:,:,:-1]  # (b n k)
        return logits  # (b n k)
