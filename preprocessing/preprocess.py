"""
Preprocess UCR-anomaly.
The dataset archive is from (https://wu.renjie.im/research/anomaly-benchmarks-are-flawed/).
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from pathlib import Path
from einops import rearrange, repeat

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal
from scipy.stats import multivariate_normal
from scipy import interpolate

from utils import get_root_dir, set_window_size


# data_dir = Path(
#     "dataset/AnomalyDatasets_2021/UCR_TimeSeriesAnomalyDatasets2021/FilesAreInHere/UCR_Anomaly_FullData/"
# )
data_dir = get_root_dir().joinpath('preprocessing', 'dataset', 'AnomalyDatasets_2021', 'UCR_TimeSeriesAnomalyDatasets2021', 'FilesAreInHere', 'UCR_Anomaly_FullData')
# print(data_dir)

pattern = re.compile(r"^([0-9]{3})_UCR_Anomaly_([a-zA-Z0-9]+)_([0-9]+)_([0-9]+)_([0-9]+).txt$")


@dataclass
class UCR_AnomalySequence:
    name: str
    id: int

    train_start: int  # starts at 1
    train_stop: int

    anom_start: int
    anom_stop: int

    data: np.ndarray

    @property
    def train_data(self):
        # not 100% sure about their indexing, this might be off by one
        # assume we include both right and left index
        return self.data[self.train_start : self.train_stop]

    @property
    def test_data(self):
        # not 100% sure about their indexing, this might be off by one
        # assume we include both right and left index
        return self.data[self.train_stop:]

    @property
    def anom_data(self):
        # not 100% sure about their indexing, this might be off by one
        # assume we include both right and left index
        return self.data[self.anom_start : self.anom_stop]

    @classmethod
    def create(cls, path: Path) -> UCR_AnomalySequence:

        assert path.exists()
        data = np.loadtxt(path, dtype=np.float32)

        match = pattern.match(path.name)
        assert match

        id, name, train_stop, anom_start, anom_stop = match.groups()

        return cls(
            name=name,
            id=int(id),
            train_start=1,
            train_stop=int(train_stop) + 1,  # +1 to make python-index easier
            anom_start=int(anom_start),
            anom_stop=int(anom_stop) + 1,  # +1 to make python-index easier
            data=data,
        )

    @classmethod
    def create_by_id(cls, id: int) -> UCR_AnomalySequence:
        return cls.create((next(data_dir.glob(f"{id:03d}_*"))))

    @classmethod
    def create_by_name(cls, name: str) -> UCR_AnomalySequence:
        return cls.create((next(data_dir.glob(f"*_UCR_Anomaly_{name}_*"))))


def scale(x, return_scale_params: bool = False):
    """
    instance-wise scaling.
    global-scaling is not used because there are time series with local-mean-shifts such as "UCR_Anomaly_sddb49_20000_67950_68200.txt".

    :param x: (n_convariates, window_length) or it can be (batch, n_convariates, window_length)
    :return: scaled x
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()

    # centering
    mu = torch.nanmean(x, dim=-1, keepdim=True)
    x = x - mu

    # var scaling
    std = torch.std(x, dim=-1, keepdim=True)  # (n_covariates, 1)
    min_std = 1.e-4  # to prevent numerical instability in scaling.
    std = torch.clamp(std, min_std, None)  # same as np.clip; (n_covariates, 1)
    x = x / std

    if return_scale_params:
        return x, (mu, std)
    else:
        return x
    

class UCRAnomalyDataset(Dataset):
    def __init__(self,
                 kind:str,
                 dataset_importer: UCR_AnomalySequence,
                 window_size:int,
                 ):
        assert kind in ['train', 'test']
        self.kind = kind
        self.window_size = window_size

        if kind == 'train':
            self.X = dataset_importer.train_data[:, None]  # add channel dim; (ts_len, 1)
        elif kind == 'test':
            self.X = dataset_importer.test_data[:, None]  # (ts_len, 1)
            # anomaly data
            self.anom_start = dataset_importer.anom_start - dataset_importer.train_stop  # relative to `test_data`
            self.anom_stop = dataset_importer.anom_stop - dataset_importer.train_stop  # relative to `test_data`
            self.Y = np.zeros_like(self.X)[:, 0]  # (ts_len,)
            self.Y[self.anom_start:self.anom_stop] = 1.

        ts_len = self.X.shape[0]
        self.dataset_len = (ts_len - 1) - window_size

    def __getitem__(self, idx):
        rng = slice(idx, idx+self.window_size)

        x = self.X[rng]  # (window_size, 1); 1 denotes channel dim.
        x = rearrange(x, 'l c -> c l')  # (1, window_size)
        x = torch.from_numpy(x).float()  # (1, window_size)
        x = scale(x).float()
        
        if self.kind == 'train':
            return x
        elif self.kind == 'test':
            y = self.Y[rng]  # (window_size,)
            y = torch.from_numpy(y).long()  # (window_size,)
            return x, y

    def __len__(self):
        return self.dataset_len
