"""
Stage 1: VQ training

run `python stage1.py`
"""
import os
from argparse import ArgumentParser
import datetime

import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from experiments.exp_stage1 import ExpStage1
from preprocessing.preprocess import UCR_AnomalySequence
from preprocessing.data_pipeline import build_data_pipeline
from utils import get_root_dir, load_yaml_param_settings, set_window_size


def load_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help="Path to the config data  file.",
                        default=get_root_dir().joinpath('configs', 'config.yaml'))
    parser.add_argument('--gpu_device_ind', nargs='+', default=[0], type=int)
    parser.add_argument('--dataset_ind', default='1', nargs='+', help='e.g., 1 2 3. Indices of datasets to run experiments on.')
    return parser.parse_args()


def train_stage1(config: dict,
                 dataset_idx: int,
                 window_size: int,
                 train_data_loader: DataLoader,
                 test_data_loader: DataLoader,
                 gpu_device_ind: list,
                 ):
    """
    :param do_validate: if True, validation is conducted during training with a test dataset.
    """
    project_name = 'TimeVQVAE-AnomalyDetection-stage1'
    
    # fit
    input_length = window_size
    train_exp = ExpStage1(input_length, config)

    n_trainable_params = sum(p.numel() for p in train_exp.parameters() if p.requires_grad)
    extra_config = {'dataset.idx': dataset_idx, 'n_trainable_params': n_trainable_params, 'gpu_device_ind': gpu_device_ind}
    wandb_logger = WandbLogger(project=project_name, name=None, config={**config, **extra_config})

    trainer = pl.Trainer(logger=wandb_logger,
                         enable_checkpointing=False,
                         callbacks=[LearningRateMonitor(logging_interval='step')],
                         max_steps=config['trainer_params']['max_steps']['stage1'],
                         devices=gpu_device_ind,
                         accelerator="gpu",
                         strategy='ddp_find_unused_parameters_true' if len(gpu_device_ind) > 1 else "auto",
                         val_check_interval=config['trainer_params']['val_check_interval']['stage1'],
                         check_val_every_n_epoch=None,
                         max_time=datetime.timedelta(hours=config['trainer_params']['max_hours']['stage1']),
                         )
    trainer.fit(train_exp,
                train_dataloaders=train_data_loader,
                val_dataloaders=test_data_loader,
                )

    # test
    print('closing...')
    wandb.finish()

    print('saving the models...')
    trainer.save_checkpoint(os.path.join(f'saved_models', f'stage1-{dataset_idx}.ckpt'))


if __name__ == '__main__':
    # load config
    args = load_args()
    config = load_yaml_param_settings(args.config)

    for idx in args.dataset_ind:
        dataset_idx = int(idx)
        dataset_importer = UCR_AnomalySequence.create_by_id(dataset_idx)

        window_size = set_window_size(dataset_idx, config['dataset']['n_periods'])
        batch_size = config['dataset']['batch_sizes']['stage1']
        num_workers = config['dataset']["num_workers"]
        train_data_loader, test_data_loader = [build_data_pipeline(batch_size,
                                                                   dataset_importer,
                                                                   kind,
                                                                   window_size,
                                                                   num_workers) for kind in ['train', 'test']]

        # train
        train_stage1(config, dataset_idx, window_size, train_data_loader, test_data_loader, args.gpu_device_ind)
