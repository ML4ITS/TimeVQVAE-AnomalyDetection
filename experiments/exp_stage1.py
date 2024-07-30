import einops
import torch
import numpy as np
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import wandb
import pytorch_lightning as pl

from models.stage1.vq_vae_encdec import VQVAEEncoder, VQVAEDecoder
from models.stage1.vq import VectorQuantize
from utils import compute_downsample_rate, timefreq_to_time, time_to_timefreq, quantize, linear_warmup_cosine_annealingLR


class ExpStage1(pl.LightningModule):
    def __init__(self,
                 input_length: int,
                 config: dict):
        """
        :param input_length: length of input time series
        :param config: configs/config.yaml
        :param n_train_samples: number of training samples
        """
        super().__init__()
        self.config = config
        self.n_fft = config['VQ-VAE']['n_fft']
        dim = config['encoder']['dim']
        in_channels = config['dataset']['in_channels']
        downsampled_width = config['encoder']['downsampled_width']
        downsample_rate = compute_downsample_rate(input_length, self.n_fft, downsampled_width)

        self.encoder = VQVAEEncoder(dim, 2*in_channels, downsample_rate, config['encoder']['n_resnet_blocks'], self.n_fft, frequency_indepence=True)
        self.vq_model = VectorQuantize(dim, **config['VQ-VAE'])
        self.decoder = VQVAEDecoder(dim, 2*in_channels, downsample_rate, config['decoder']['n_resnet_blocks'], input_length, self.n_fft, in_channels, frequency_indepence=True)

    def forward(self, batch_idx:int, x, y=None):
        """
        :param x: input time series (b c l)
        """

        z = self.encoder(x)
        z_q, s, vq_loss, perplexity = quantize(z, self.vq_model)
        x_rec = self.decoder(z_q)  # (b c l)
        recons_loss = F.mse_loss(x, x_rec)

        # plot `x` and `x_rec`
        if not self.training and batch_idx==0:
            b = np.random.randint(0, x.shape[0])
            c = np.random.randint(0, x.shape[1])

            fig, ax = plt.subplots(1, 1, figsize=(4, 1*3))
            plt.suptitle(f'step-{self.global_step}')
            ax.plot(x[b, c].cpu(), label=r'$x$')
            ax.plot(x_rec[b, c].detach().cpu(), label=r'$x_{rec}$')
            ax.set_ylim(-4, 4)
            plt.tight_layout()
            plt.legend(loc='upper right')
            self.logger.log_image(key=f"recons ({'train' if self.training else 'val'})", images=[wandb.Image(plt),])
            plt.close()

        return recons_loss, vq_loss['loss'], perplexity
    
    def training_step(self, batch, batch_idx):
        self.train()
        x = batch
        recons_loss, vq_loss, perplexity = self.forward(batch_idx, x)
        loss = recons_loss + vq_loss

        # lr scheduler
        sch = self.lr_schedulers()
        sch.step()

        # log
        loss_hist = {'loss': loss,
                     'recons_loss': recons_loss,
                     'vq_loss': vq_loss,
                     'perplexity': perplexity,
                     }
        for k, v in loss_hist.items():
            self.log(f'train/{k}', v)
        
        return loss_hist

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.eval()  # Set the model to evaluation mode
        x, y = batch
        recons_loss, vq_loss, perplexity = self.forward(batch_idx, x, y=y)
        loss = recons_loss + vq_loss

        # log
        loss_hist = {'loss': loss,
                     'recons_loss': recons_loss,
                     'vq_loss': vq_loss,
                     'perplexity': perplexity,
                     }
        for k, v in loss_hist.items():
            self.log(f'val/{k}', v)

        return loss_hist

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.config['exp_params']['lr']['stage1'])
        scheduler = linear_warmup_cosine_annealingLR(opt, self.config['trainer_params']['max_steps']['stage1'], self.config['exp_params']['linear_warmup_rate'])
        return {'optimizer': opt, 'lr_scheduler': scheduler}
