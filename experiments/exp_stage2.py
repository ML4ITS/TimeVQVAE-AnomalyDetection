import matplotlib.pyplot as plt
import torch.nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl

from models.stage2.maskgit import MaskGIT
from utils import linear_warmup_cosine_annealingLR


class ExpStage2(pl.LightningModule):
    def __init__(self,
                 dataset_idx: int,
                 input_length: int,
                 config: dict):
        super().__init__()
        self.config = config
        self.maskgit = MaskGIT(dataset_idx, input_length, **config['MaskGIT'], config=config)

    def forward(self, x):
        """
        :param x: (B, C, L)
        """
        pass

    def training_step(self, batch, batch_idx):
        self.train()
        x = batch
        device = x.device

        mask_pred_loss = self.maskgit(x)
        loss = mask_pred_loss

        # lr scheduler
        sch = self.lr_schedulers()
        sch.step()

        # log
        loss_hist = {'loss': loss,
                     'mask_pred_loss': mask_pred_loss,}
        for k, v in loss_hist.items():
            self.log(f'train/{k}', v)

        return loss_hist

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.eval()  # set the model on the evaluation mode

        x, y = batch
        device = x.device

        mask_pred_loss = self.maskgit(x)
        loss = mask_pred_loss

        # log
        loss_hist = {'loss': loss,
                     'mask_pred_loss': mask_pred_loss,}
        for k, v in loss_hist.items():
            self.log(f'val/{k}', v)

        if batch_idx == 0:
            self.maskgit.eval()

            # unconditional sampling
            s = self.maskgit.iterative_decoding(device=device)
            xhat = self.maskgit.decode_token_ind_to_timeseries(s).cpu()

            b = 0
            fig, axes = plt.subplots(2, 1, figsize=(4, 1.5*2))
            plt.suptitle(f'step-{self.global_step}')
            axes[0].set_title(r'$\hat{x}$')
            axes[0].plot(xhat[b, 0, :])
            for ax in axes:
                ax.set_ylim(-4, 4)
            plt.tight_layout()
            self.logger.log_image(key=f"generated sample ({'train' if self.training else 'val'})", images=[wandb.Image(plt),])
            plt.close()

        return loss_hist

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.config['exp_params']['lr']['stage2'])
        scheduler = linear_warmup_cosine_annealingLR(opt, self.config['trainer_params']['max_steps']['stage2'], self.config['exp_params']['linear_warmup_rate'])
        return {'optimizer': opt, 'lr_scheduler': scheduler}