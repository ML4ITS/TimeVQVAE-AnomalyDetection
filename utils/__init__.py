import os
import pickle
import logging
import yaml
import tempfile
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit
from torch.optim.lr_scheduler import SequentialLR, LambdaLR, CosineAnnealingLR
import numpy as np
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from scipy.signal import argrelextrema
import pandas as pd


def get_root_dir():
    return Path(__file__).parent.parent
# root_dir = Path(__file__).parent.parent
prefix = os.path.join('datasets', 'processed')


def load_yaml_param_settings(yaml_fname: str):
    """
    :param yaml_fname: .yaml file that consists of hyper-parameter settings.
    """
    stream = open(yaml_fname, 'r')
    config = yaml.load(stream, Loader=yaml.FullLoader)
    return config


def preprocess(df, scaler: MinMaxScaler, kind: str):
    """returns normalized and standardized data.
    """

    df = np.asarray(df, dtype=np.float32)

    if len(df.shape) == 1:
        raise ValueError('Data must be a 2-D array')

    if np.any(sum(np.isnan(df)) != 0):
        print('Data contains null values. Will be replaced with 0')
        df = np.nan_to_num()

    # normalize data (corrected)
    if kind == 'train':
        df = scaler.fit_transform(df)
    elif kind == 'test':
        df = scaler.transform(df)
    # df = MinMaxScaler().fit_transform(df)  # -> previous incorrect scaling method
    print('Data normalized')

    return df


def minibatch_slices_iterator(length, step_size,
                              ignore_incomplete_batch=False):
    """
    Iterate through all the mini-batch slices.

    Args:
        length (int): Total length of data in an epoch.
        step_size (int):
        ignore_incomplete_batch (bool): If :obj:`True`, discard the final
            batch if it contains less than `batch_size` number of items.
            (default :obj:`False`)

    Yields
        slice: Slices of each mini-batch.  The last mini-batch may contain
               less indices than `batch_size`.
    """
    start = 0
    stop1 = (length // step_size) * step_size
    while start < stop1:
        yield slice(start, start + step_size, 1)
        start += step_size
    if not ignore_incomplete_batch and start < length:
        yield slice(start, length, 1)


class BatchSlidingWindow(object):
    """
    Class for obtaining mini-batch iterators of sliding windows.

    Each mini-batch will have `batch_size` windows.  If the final batch
    contains less than `batch_size` windows, it will be discarded if
    `ignore_incomplete_batch` is :obj:`True`.

    Args:
        array_size (int): Size of the arrays to be iterated.
        window_size (int): The size of the windows.
        batch_size (int): Size of each mini-batch.
        excludes (np.ndarray): 1-D `bool` array, indicators of whether
            or not to totally exclude a point.  If a point is excluded,
            any window which contains that point is excluded.
            (default :obj:`None`, no point is totally excluded)
        shuffle (bool): If :obj:`True`, the windows will be iterated in
            shuffled order. (default :obj:`False`)
        ignore_incomplete_batch (bool): If :obj:`True`, discard the final
            batch if it contains less than `batch_size` number of windows.
            (default :obj:`False`)
    """

    def __init__(self, array_size, window_size, step_size, batch_size, excludes=None,
                 shuffle=False, ignore_incomplete_batch=False):
        # check the parameters
        if window_size < 1:
            raise ValueError('`window_size` must be at least 1')
        if array_size < window_size:
            raise ValueError('`array_size` must be at least as large as '
                             '`window_size`')
        if excludes is not None:
            excludes = np.asarray(excludes, dtype=np.bool)
            expected_shape = (array_size,)
            if excludes.shape != expected_shape:
                raise ValueError('The shape of `excludes` is expected to be '
                                 '{}, but got {}'.
                                 format(expected_shape, excludes.shape))

        # compute which points are not excluded
        if excludes is not None:
            mask = np.logical_not(excludes)
        else:
            mask = np.ones([array_size], dtype=np.bool)
        mask[: window_size - 1] = False
        where_excludes = np.where(excludes)[0]
        for k in range(1, window_size):
            also_excludes = where_excludes + k
            also_excludes = also_excludes[also_excludes < array_size]
            mask[also_excludes] = False

        # generate the indices of window endings
        indices = np.arange(array_size)[mask]
        self._indices = indices.reshape([-1, 1])

        # the offset array to generate the windows
        self._offsets = np.arange(-window_size + 1, 1)

        # memorize arguments
        self._array_size = array_size
        self._window_size = window_size
        self._step_size = step_size
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._ignore_incomplete_batch = ignore_incomplete_batch

    def get_iterator(self, arrays):
        """
        Iterate through the sliding windows of each array in `arrays`.

        This method is not re-entrant, i.e., calling :meth:`get_iterator`
        would invalidate any previous obtained iterator.

        Args:
            arrays (Iterable[np.ndarray]): 1-D arrays to be iterated.

        Yields:
            tuple[np.ndarray]: The windows of arrays of each mini-batch.
        """
        # check the parameters
        arrays = tuple(np.asarray(a) for a in arrays)
        if not arrays:
            raise ValueError('`arrays` must not be empty')

        # shuffle if required
        if self._shuffle:
            np.random.shuffle(self._indices)

        # iterate through the mini-batches
        for s in minibatch_slices_iterator(
                length=len(self._indices),
                step_size=self._step_size,
                ignore_incomplete_batch=self._ignore_incomplete_batch):
            idx = self._indices[s] + self._offsets
            yield tuple(a[idx] if len(a.shape) == 1 else a[idx, :] for a in arrays)


def freeze(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True


def save_model(models_dict: dict, dirname='saved_models', id = ''):
    """
    :param models_dict: {'model_name': model, ...}
    """
    try:
        if not os.path.isdir(get_root_dir().joinpath(dirname)):
            os.mkdir(get_root_dir().joinpath(dirname))

        id = str(id)
        id_ = id[:]
        if id != '':
            id_ = '-' + id_
        for model_name, model in models_dict.items():
            torch.save(model.state_dict(), get_root_dir().joinpath(dirname, model_name + id_ + '.ckpt'))
    except PermissionError:
        # dirname = tempfile.mkdtemp()
        dirname = tempfile.gettempdir()
        print(f'\nThe trained model is saved in the following temporary dirname due to some permission error: {dirname}.\n')

        id_ = id[:]
        if id != '':
            id_ = '-' + id_
        for model_name, model in models_dict.items():
            torch.save(model.state_dict(), get_root_dir().joinpath(dirname, model_name + id_ + '.ckpt'))


def time_to_timefreq(x, n_fft: int, C: int):
    """
    x: (B, C, L)
    """
    x = rearrange(x, 'b c l -> (b c) l')
    x = torch.stft(x, n_fft, normalized=True, return_complex=True, window=torch.hann_window(window_length=n_fft, device=x.device))
    x = torch.view_as_real(x)  # (B, H, W, 2); 2: (real, imag)
    x = rearrange(x, '(b c) n t z -> b (c z) n t ', c=C)  # z=2 (real, imag)
    return x  # (B, C, H, W)


def timefreq_to_time(x, n_fft: int, C: int):
    x = rearrange(x, 'b (c z) n t -> (b c) n t z', c=C)
    x = x.contiguous()
    x = torch.view_as_complex(x)
    x = torch.istft(x, n_fft, normalized=True, window=torch.hann_window(window_length=n_fft, device=x.device))
    x = rearrange(x, '(b c) l -> b c l', c=C)
    return x


# def compute_var_loss(z):
#     return torch.relu(1. - torch.sqrt(z.var(dim=0) + 1e-4)).mean()
#
#
# def compute_cov_loss(z):
#     norm_z = (z - z.mean(dim=0))
#     norm_z = F.normalize(norm_z, p=2, dim=0)  # (batch * feature); l2-norm
#     fxf_cov_z = torch.mm(norm_z.T, norm_z)  # (feature * feature)
#     ind = np.diag_indices(fxf_cov_z.shape[0])
#     fxf_cov_z[ind[0], ind[1]] = torch.zeros(fxf_cov_z.shape[0]).to(norm_z.device)
#     cov_loss = (fxf_cov_z ** 2).mean()
#     return cov_loss


# def quantize(z, vq_model, transpose_channel_length_axes=False, return_prob: bool = False):
#     input_dim = len(z.shape) - 2
#     if input_dim == 2:
#         h, w = z.shape[2:]
#         z = rearrange(z, 'b c h w -> b (h w) c')
#         if return_prob:
#             z_q, indices, vq_loss, perplexity, (dist, prob) = vq_model(z, return_prob)
#         else:
#             z_q, indices, vq_loss, perplexity = vq_model(z, return_prob)
#         z_q = rearrange(z_q, 'b (h w) c -> b c h w', h=h, w=w)
#     elif input_dim == 1:
#         if transpose_channel_length_axes:
#             z = rearrange(z, 'b c l -> b (l) c')
#         if return_prob:
#             z_q, indices, vq_loss, perplexity, (dist, prob) = vq_model(z, return_prob)
#         else:
#             z_q, indices, vq_loss, perplexity = vq_model(z, return_prob)
#         if transpose_channel_length_axes:
#             z_q = rearrange(z_q, 'b (l) c -> b c l')
#     else:
#         raise ValueError
#
#     if return_prob:
#         return z_q, indices, vq_loss, perplexity, (dist, prob)
#     else:
#         return z_q, indices, vq_loss, perplexity


def quantize(z, vq_model, transpose_channel_length_axes=False, svq_temp=None):
    input_dim = len(z.shape) - 2
    if input_dim == 2:
        h, w = z.shape[2:]
        z = rearrange(z, 'b c h w -> b (h w) c')
        z_q, indices, vq_loss, perplexity = vq_model(z, svq_temp=svq_temp)
        z_q = rearrange(z_q, 'b (h w) c -> b c h w', h=h, w=w)
    elif input_dim == 1:
        if transpose_channel_length_axes:
            z = rearrange(z, 'b c l -> b (l) c')
        z_q, indices, vq_loss, perplexity = vq_model(z, svq_temp=svq_temp)
        if transpose_channel_length_axes:
            z_q = rearrange(z_q, 'b (l) c -> b c l')
    else:
        raise ValueError
    return z_q, indices, vq_loss, perplexity


def zero_pad_high_freq(xf):
    """
    xf: (B, C, H, W); H: frequency-axis, W: temporal-axis
    """
    xf_l = torch.zeros(xf.shape).to(xf.device)
    xf_l[:, :, 0, :] = xf[:, :, 0, :]
    return xf_l


def zero_pad_low_freq(xf):
    """
    xf: (B, C, H, W); H: frequency-axis, W: temporal-axis
    """
    # if kind == 'target':
    #     xf_h = torch.zeros(xf.shape).to(xf.device)
    #     xf_h[:, :, 1:, :] = xf[:, :, 1:, :]
    #     return xf_h
    # elif kind == 'predict':
    #     xfhat_h = torch.zeros(xf.shape).to(xf.device)
    #     xfhat_h[:, :, 1:, :] = xfhat
    #     return xfhat_h
    xf_h = torch.zeros(xf.shape).to(xf.device)
    xf_h[:, :, 1:, :] = xf[:, :, 1:, :]
    return xf_h

def compute_emb_loss(codebook, x, use_cosine_sim, esm_max_codes):
    embed = codebook.embed
    flatten = x.reshape(-1, x.shape[-1])

    if use_cosine_sim:
        flatten = F.normalize(flatten, p=2, dim=-1)
        embed = F.normalize(embed, p=2, dim=-1)

    # N samples can be sampled fro embed for the memory efficiency.
    ind = torch.randint(0, embed.shape[0], size=(min(esm_max_codes, embed.shape[0]),))
    embed = embed[ind]

    cov_embed = torch.cov(embed.t())  # (D, D)
    cov_x = torch.cov(flatten.t())  # (D, D)

    mean_embed = torch.mean(embed, dim=0)
    mean_x = torch.mean(flatten, dim=0)

    esm_loss = F.mse_loss(cov_x.detach(), cov_embed) + F.mse_loss(mean_x.detach(), mean_embed)
    return esm_loss


def compute_downsample_rate(input_length: int,
                            n_fft: int,
                            downsampled_length: int):
    return round(input_length / (np.log2(n_fft) - 1) / downsampled_length) if input_length >= downsampled_length else 1


# def set_window_size(train_data, min_window_size: int = 200, max_window_size_rate: float = 0.3, min_acf: float = 0.3):
#     """
#     :param train_data: (length,)
#     :param min_acf: minimum ACF value
#     :return window_size
#     """
#     acf_train = sm.tsa.acf(train_data, nlags=train_data.shape[0] - 1)[:train_data.shape[0] // 4]
#     local_maxima = argrelextrema(acf_train, np.greater)[0]
#     local_maxima = np.array([i for i in local_maxima if acf_train[i] > min_acf])
#
#     if len(local_maxima) > 0:
#         cycle_length = local_maxima[acf_train[local_maxima].argmax()]
#         window_size = 5 * cycle_length if len(local_maxima) > 0 else min_window_size
#         max_window_size = int(max_window_size_rate * train_data.shape[0])
#         window_size = np.clip(window_size, min_window_size, max_window_size)
#     else:
#         window_size = min_window_size
#     return window_size


def set_window_size(dataset_id, n_periods: int):
    period_data = get_root_dir().joinpath('preprocessing', 'UCR_anomaly_dataset_periods.csv')
    df = pd.read_csv(str(period_data))
    single_period = int(df.query(f"dataset_idx == {dataset_id}")['period'].item())
    return single_period * n_periods


def compute_receptive_field(layers):
    """
    layers: [(kernel_size, stride_size, padding_size), (kernel_size, stride_size, padding_size), ...]
    """
    receptive_field = 1
    stride_product = 1

    for (kernel_size, stride_size, padding_size) in reversed(layers):
        stride_product *= stride_size
        receptive_field += (kernel_size - 1) * stride_product
    return receptive_field


def linear_warmup_cosine_annealingLR(optimizer: torch.optim.Optimizer, max_steps:int, linear_warmup_rate:float=0.1, min_lr:float=1e-6):
    assert linear_warmup_rate > 0. and linear_warmup_rate < 1., '0 < linear_warmup_rate < 1.'

    warmup_steps = int(max_steps * linear_warmup_rate)  # n% of max_steps

    # Define the warmup scheduler
    def warmup_lambda(current_step):
        if current_step >= warmup_steps:
            return 1.0
        return float(current_step) / float(max(1, warmup_steps))

    # Create the warmup scheduler
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)

    # Create the cosine annealing scheduler
    cosine_scheduler = CosineAnnealingLR(optimizer, max_steps - warmup_steps, eta_min=min_lr)

    # Combine the warmup and cosine annealing schedulers
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
    return scheduler


class SnakeActivation(jit.ScriptModule):
    """
    this version allows multiple values of `a` for different channels/num_features
    """
    def __init__(self, num_features:int, dim:int, a_base=0.2, learnable=True, a_max=0.5):
        super().__init__()
        assert dim in [1, 2], '`dim` supports 1D and 2D inputs.'

        if learnable:
            if dim == 1:  # (b d l); like time series
                a = np.random.uniform(a_base, a_max, size=(1, num_features, 1))  # (1 d 1)
                self.a = nn.Parameter(torch.tensor(a, dtype=torch.float32))
            elif dim == 2:  # (b d h w); like 2d images
                a = np.random.uniform(a_base, a_max, size=(1, num_features, 1, 1))  # (1 d 1 1)
                self.a = nn.Parameter(torch.tensor(a, dtype=torch.float32))
        else:
            self.register_buffer('a', torch.tensor(a_base, dtype=torch.float32))

    @jit.script_method
    def forward(self, x):
        return x + (1 / self.a) * torch.sin(self.a * x) ** 2