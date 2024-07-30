"""
compute the accuracy, as suggested in [1]

[1] Wu, Renjie, and Eamonn Keogh. "Current time series anomaly detection benchmarks are flawed and are creating the illusion of progress." IEEE Transactions on Knowledge and Data Engineering (2021).
"""
import os
from argparse import ArgumentParser
import copy
import pickle
import gc

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange, repeat
import torch.nn.functional as F
import scipy.stats as stats
from scipy.signal import find_peaks

from experiments.exp_stage2 import ExpStage2
from preprocessing.preprocess import scale
from utils import get_root_dir, set_window_size
from preprocessing.preprocess import UCR_AnomalySequence, UCRAnomalyDataset
from models.stage2.maskgit import MaskGIT


def load_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help="Path to the config data  file.",
                        default=get_root_dir().joinpath('configs', 'config.yaml'))
    return parser.parse_args([])


def load_data(dataset_idx, config, kind: str):
    """used during evaluation"""
    assert kind in ['train', 'test']

    dataset_importer = UCR_AnomalySequence.create_by_id(dataset_idx)
    window_size = set_window_size(dataset_idx, config['dataset']['n_periods'])
    dataset = UCRAnomalyDataset(kind, dataset_importer, window_size)

    X = dataset.X  # (ts_len, 1)
    X = rearrange(X, 'l c -> c l')  # (1, ts_len)

    if kind == 'train':
        return X
    elif kind == 'test':
        Y = dataset.Y  # (ts_len,)
        return X, Y


def mask_prediction(s, height, slice_rng, maskgit:MaskGIT):
    # mask
    s_m = copy.deepcopy(s)  # (1 n)
    s_m = rearrange(s_m, '1 (h w) -> 1 h w', h=height)  # (1 h w)
    s_m[:, :, slice_rng] = maskgit.mask_token_id

    # mask-prediction
    logits = maskgit.transformer(rearrange(s_m, 'b h w -> b (h w)'))  # (1 n K)
    logits = rearrange(logits, '1 (h w) K -> 1 h w K', h=height)  # (1 h w K)
    logits_prob = F.softmax(logits, dim=-1)  # (1 h w K)

    return logits, logits_prob




# def zq_to_z_shape(zq):
#     """
#     zq: (b d h' w')
#     """
#     b, d, h, w = zq.shape
#     rope = nn.Sequential(Rearrange('b d n -> b n 1 d'),
#                          RotaryPositionalEmbeddings(dim=d),
#                          Rearrange('b n 1 d -> b d n'))
    
#     zq = rearrange(zq, 'b d h w -> b d (h w)')  # (b d n)
#     z_shape = rope(zq).mean(dim=-1)  # (b d)  # (b d)
#     return z_shape  # (b d)


@torch.no_grad()
def detect(X_unscaled,
           maskgit:MaskGIT,
           window_size: int,
           rolling_window_stride: int,
           latent_window_size: int,
           compute_reconstructed_X: bool,
           device: int,
           ):
    """
    :param X_unscaled: (1, ts_len)
    :param maskgit:
    :param window_size:
    :param rolling_window_stride:
    :param latent_window_size:
    :param compute_reconstructed_X:
    :return:
    """
    assert latent_window_size % 2 == 1  # latent_window_size must be an odd number.

    ts_len = X_unscaled.shape[1]
    n_channels = X_unscaled.shape[0]
    end_time_step = (ts_len - 1) - window_size
    timestep_rng = range(0, end_time_step, rolling_window_stride)
    logs = {'a_star': np.zeros((n_channels, maskgit.H_prime, ts_len)),
            'reconsX': np.zeros((n_channels, ts_len)),
            'count': np.zeros((n_channels, ts_len)),
            'last_window_rng': None,
            }
    for timestep_idx, timestep in enumerate(timestep_rng):
        if timestep_idx % int(0.3 * len(timestep_rng)) == 0:
            print(f'timestep/total_time_steps*100: {timestep}/{end_time_step} | {round(timestep / end_time_step * 100, 2)} [%]')

        # fetch a window at each timestep
        window_rng = slice(timestep, timestep+window_size)
        x_unscaled = X_unscaled[None, :, window_rng]  # (1, 1, window_size)
        x, (mu, sigma) = scale(x_unscaled, return_scale_params=True)

        # encode
        z_q, s = maskgit.encode_to_z_q(x.to(device), maskgit.encoder, maskgit.vq_model)  # s: (1 n)
        latent_height, latent_width = z_q.shape[2], z_q.shape[3]

        # compute the anomaly scores (negative log likelihood, notated as nll)
        a_tilde = np.zeros((n_channels, latent_height, latent_width),)  # (1 h w)
        for w in range(latent_width):
            kernel_rng = slice(w, w + 1) if latent_window_size == 1 else slice(max(0, w - (latent_window_size - 1) // 2), w + (latent_window_size - 1) // 2 + 1)

            # mask-prediction
            logits, logits_prob = mask_prediction(s, latent_height, kernel_rng, maskgit)  # (1 h w K)

            # prior-based anomaly score
            s_rearranged = rearrange(s, '1 (h w) -> 1 h w', h=latent_height)
            p = torch.gather(logits_prob, -1, s_rearranged[:, :, :, None])  # (1 h w 1)
            p = p[:, :, kernel_rng, 0]  # (1 h r)
            a_w = -1 * torch.log(p + 1e-30).mean(dim=-1)  # (1 h); 1e-30 for numerical stability.
            a_w = a_w.detach().cpu().numpy()  # (1 h)
            a_tilde[:, :, w] = a_w
        a_tilde_m = F.interpolate(torch.from_numpy(a_tilde[None,:,:,:]), size=(latent_height, window_size), mode='nearest')[0].numpy()  # (1 h window_size)
        logs['a_star'][:, :, window_rng] += a_tilde_m

        # reconstructed X
        if compute_reconstructed_X:
            x_recons = maskgit.decode_token_ind_to_timeseries(s).cpu().numpy()  # (1 1 window_size)
            x_recons = (x_recons * sigma.numpy()) + mu.numpy()  # (1 1 window_size)
            x_recons = F.interpolate(torch.from_numpy(x_recons), size=(window_size,), mode='linear').numpy()  # (1 1 window_size)
            logs['reconsX'][:, window_rng] += x_recons[0]

        # log per window
        logs['count'][:, window_rng] += 1

    # resulting log
    logs['last_window_rng'] = window_rng
    logs['count'] = np.clip(logs['count'], 1, None)  # to prevent zero division.
    # logs['a_star'] = logs['a_star'] / logs['count'][:, None, :]
    logs['reconsX'] = logs['reconsX'] / logs['count']
    logs['timestep_rng'] = timestep_rng

    return logs


def postprocess_XY(X):
    """
    X|Y: (n_rollowing_window_steps, n_channels, window_size)
    return X|Y_flat (b, ts_len)|
    """
    first_steps_at_every_window = X[:, 0, 0]
    last_window = X[-1, 0, 1:]
    X_pp = np.concatenate((first_steps_at_every_window, last_window), axis=-1)
    return X_pp


def postprocess_dist(dist, window_size, X_test_unscaled_pp, H_prime, latent_timesteps):
    """
    dist: (rolling_window_time_steps, latent_height, latent_width)
    """
    ts_length = X_test_unscaled_pp.shape[0]
    dist_interp = F.interpolate(torch.from_numpy(dist)[None, :, :],
                                size=(H_prime, window_size),
                                mode='nearest')[0].numpy()
    dist_pp = np.ones((dist_interp.shape[1], ts_length)) * dist_interp.min()  # (height, ts_len)
    dist_pp_count = np.ones_like(dist_pp)
    for i, t in enumerate(latent_timesteps):
        s = dist_interp[i]  # (height, window_size)
        dist_pp[:, t:t + window_size] += s
        dist_pp_count[:, t:t + window_size] += 1
    dist_pp /= dist_pp_count  # (height, ts_len)

    # reduce the height dim
    # dist_pp_reduced = dist_pp.mean(axis=0)  # (ts_len,)
    # k = int(np.ceil(dist_pp.shape[0] * 0.3))
    # dist_pp_reduced = torch.topk(torch.from_numpy(dist_pp), k=k, dim=0).values  # (reduced_height, ts_len)
    # dist_pp_reduced = dist_pp_reduced.mean(dim=0).numpy()  # (ts_len,)

    # return dist_pp, dist_pp_reduced
    return dist_pp


def postprocess_Xhat(Xhat, rolling_window_stride):
    """
    Xhat: (rolling window timesteps, n_channels, window_size)
    """
    # part 1
    Xhat_pp = {}
    total_rolling_window_steps = Xhat.shape[0]
    for timestep in range(0, total_rolling_window_steps, rolling_window_stride):
        xhat = Xhat[[timestep]]  # (1, n_channels, window_size)
        window_size = xhat.shape[-1]
        for l in range(window_size):
            Xhat_pp.setdefault(timestep + l, [])
            Xhat_pp[timestep + l].append(xhat[0, 0, l])

    # part 2
    idx_middle = len(Xhat_pp) // 2  # to select the timestep with the largest list length
    max_list_len = len(Xhat_pp[idx_middle])
    Xhat_pp_new = {i: [] for i in range(max_list_len)}
    for timestep in Xhat_pp.keys():
        for i in range(max_list_len):
            try:
                Xhat_pp_new[i].append(Xhat_pp[timestep][i])
            except IndexError:
                Xhat_pp_new[i].append(np.nan)

    for i in range(max_list_len):
        Xhat_pp_new[i] = np.array(Xhat_pp_new[i])
    Xhat_pp = Xhat_pp_new

    return Xhat_pp


def compute_latent_window_size(latent_width, latent_window_size_rate):
    latent_window_size = latent_width * latent_window_size_rate
    if np.floor(latent_window_size) == 0:
        latent_window_size = 1
    elif np.floor(latent_window_size) % 2 != 0:
        latent_window_size = int(np.floor(latent_window_size))
    elif np.ceil(latent_window_size) % 2 != 0:
        latent_window_size = int(np.ceil(latent_window_size))
    elif latent_window_size % 2 == 0:
        latent_window_size = int(latent_window_size + 1)
    else:
        raise ValueError
    return latent_window_size


def evaluate_fn(config,
                dataset_idx: int,
                latent_window_size_rate: float,
                rolling_window_stride_rate: float,
                q: float,
                device: int = 0):
    """
    @ settings
    - device: gpu device index
    - dataset_idx: dataset index
    - rolling_window_stride_rate: stride = rolling_window_stride_rate * window_size
    - latent_window_size_rate: latent_window_size = latent_window_size_rate * latent_window_size (i.e., latent width)
    """
    # load model
    input_length = window_size = set_window_size(dataset_idx, config['dataset']['n_periods'])
    stage2 = ExpStage2.load_from_checkpoint(os.path.join('saved_models', f'stage2-{dataset_idx}.ckpt'), 
                                            dataset_idx=dataset_idx, input_length=input_length, config=config, 
                                            map_location=f'cuda:{device}')
    maskgit = stage2.maskgit
    maskgit.eval()

    # kernel size (needs to be an odd number just like for the convolutional layers.)
    rolling_window_stride = round(window_size * rolling_window_stride_rate)
    latent_window_size = compute_latent_window_size(maskgit.W_prime.item(), latent_window_size_rate)

    # compute the anomaly score on the training set
    print('===== compute the anomaly scores of the training set... =====')
    X_train_unscaled = load_data(dataset_idx, config, 'train')  # (1, ts_len)
    logs_train = detect(X_train_unscaled,
                        maskgit,
                        window_size,
                        rolling_window_stride,
                        latent_window_size,
                        compute_reconstructed_X=False,
                        device=device)

    # anomaly threshold
    if q <= 1.0:
        anom_threshold = np.quantile(logs_train['a_star'], q=q, axis=-1)  # (n_channels H')
    else:
        anom_threshold = np.quantile(logs_train['a_star'], q=1.0, axis=-1)  # (n_channels H')
        anom_threshold += anom_threshold * (q - 1.0)

    print('===== compute the anomaly scores of the test set... =====')
    X_test_unscaled, Y = load_data(dataset_idx, config, 'test')  # (1, ts_len), (ts_len,)
    logs_test = detect(X_test_unscaled,
                       maskgit,
                       window_size,
                       rolling_window_stride,
                       latent_window_size,
                       compute_reconstructed_X=True,
                       device=device)

    # clip up to the last timestep
    X_test_unscaled = X_test_unscaled[:, :logs_test['last_window_rng'].stop]
    a_star = logs_test['a_star'][:, :, :logs_test['last_window_rng'].stop]
    X_recons_test = logs_test['reconsX'][:, :logs_test['last_window_rng'].stop]
    Y = Y[:logs_test['last_window_rng'].stop]

    # anomaly score
    # univariate time series; choose the first channel
    X_test_unscaled = X_test_unscaled[0]  # (ts_len')
    a_star = a_star[0]  # (n_freq, ts_len')
    X_recons_test = X_recons_test[0]  # (ts_len')
    anom_threshold = anom_threshold[0]  # univariate time series; (n_freq,)

    # ================================ plot ================================
    n_rows = 6
    fig, axes = plt.subplots(n_rows, 1, figsize=(25, 1.5 * n_rows))
    fontsize= 15

    # plot: X_test & labels
    i = 0
    axes[i].plot(X_test_unscaled, color='black')
    axes[i].set_xlim(0, X_test_unscaled.shape[0] - 1)
    axes[i].set_title(f"dataset idx: {dataset_idx} | latent window size rate: {latent_window_size_rate}", fontsize=20)
    ax2 = axes[i].twinx()
    ax2.plot(Y, alpha=0.5, color='C1')

    # plot: anomaly score
    i += 1
    vmin = np.nanquantile(np.array(a_star).flatten(), q=0.5)  # q to remove the insignificant values in imshow.
    axes[i].imshow(a_star, interpolation='nearest', aspect='auto', vmin=vmin)
    axes[i].invert_yaxis()
    axes[i].set_xticks([])
    ylabel = 'clipped\n' + r'$a^*$'
    axes[i].set_ylabel(ylabel, fontsize=fontsize, rotation=0, labelpad=10, ha='right', va='center')

    ylim_max = np.max(a_star) * 1.05
    for j in range(a_star.shape[0]):
        i += 1
        axes[i].plot(a_star[j])
        axes[i].set_xticks([])
        xlim = (0, a_star[j].shape[0] - 1)
        axes[i].set_xlim(*xlim)
        axes[i].set_ylim(None, ylim_max)
        # axes[i].set_ylim(0, 60)
        # axes[i].set_ylabel(f'anom (freq={j + 1}-th) \n kernel_size: {kernel_size}')
        h_idx = f'h={j}'
        axes[i].set_ylabel(r'$(a^*)_{{{}}}$'.format(h_idx),
                           fontsize=fontsize, rotation=0, labelpad=35, va='center')
        threshold = 1e99 if anom_threshold[j] == np.inf else anom_threshold[j]
        axes[i].hlines(threshold, xmin=xlim[0], xmax=xlim[1], linestyle='--', color='black')

    # plot: reconstruction
    i += 1
    axes[i].plot(X_test_unscaled, color='black')
    axes[i].plot(X_recons_test, alpha=0.5, color='C1')
    axes[i].set_xticks([])
    axes[i].set_xlim(0, X_test_unscaled.shape[0] - 1)
    axes[i].set_ylabel('recons', fontsize=fontsize)

    # save: plot
    plt.tight_layout()
    plt.savefig(get_root_dir().joinpath('evaluation', 'results', f'{dataset_idx}-anomaly_score-latent_window_size_rate_{latent_window_size_rate}.png'))
    plt.close()

    # save: resulting data
    resulting_data = {'dataset_index': dataset_idx,
                      'latent_window_size_rate': latent_window_size_rate,
                      'latent_window_size': latent_window_size,
                      'rolling_window_stride_rate': rolling_window_stride_rate,
                      'q': q,

                      'X_test_unscaled': X_test_unscaled,
                      'Y': Y,
                      'a_star': a_star,
                      'X_recons_test': X_recons_test,

                      'timestep_rng_test': logs_test['timestep_rng'],
                      'anom_threshold': anom_threshold,  # (n_freq,)
                      }

    saving_fname = get_root_dir().joinpath('evaluation', 'results', f'{dataset_idx}-anomaly_score-latent_window_size_rate_{latent_window_size_rate}.pkl')
    with open(str(saving_fname), 'wb') as f:
        pickle.dump(resulting_data, f, pickle.HIGHEST_PROTOCOL)


def save_final_summarized_figure(dataset_idx, X_test_unscaled, Y, timestep_rng_test,
                                 a_s_star, a_bar_s_star, a_2bar_s_star, a_final,
                                 joint_threshold, final_threshold, anom_ind,
                                 window_size, config, args):
    n_rows = 9
    fig, axes = plt.subplots(n_rows, 1, figsize=(25, 1.5 * n_rows))
    fontsize= 15

    # plot: X_test & labels
    i = 0
    axes[i].plot(X_test_unscaled, color='black')
    axes[i].set_xlim(0, X_test_unscaled.shape[0] - 1)
    axes[i].set_title(f"dataset idx: {dataset_idx}", fontsize=fontsize)
    ax2 = axes[i].twinx()
    ax2.plot(Y, alpha=0.5, color='C1')

    # plot (imshow): a_s^*
    i += 1
    a_s_star_clipped = np.copy(a_s_star)
    a_s_star_clipped[:, ~anom_ind] = 0. if anom_ind.mean() == 0 else np.min(a_s_star_clipped[:, anom_ind])
    axes[i].imshow(a_s_star_clipped, interpolation='nearest', aspect='auto', cmap='jet')  # , vmin=vmin)
    axes[i].invert_yaxis()
    axes[i].set_xticks([])
    ylabel = 'clipped\n' + r'$a_s^*$'
    axes[i].set_ylabel(ylabel, fontsize=fontsize, rotation=0, labelpad=10, ha='right', va='center')

    # plot: a_s^*
    n_freq = a_s_star.shape[0]
    max_anom = a_s_star.max()
    for j in range(n_freq):
        i += 1
        axes[i].plot(a_s_star[j], color='lightsalmon')
        axes[i].set_xticks([])
        axes[i].set_xlim(0, a_s_star.shape[1] - 1)
        h_idx = f'h={j}'
        axes[i].set_ylabel(r'$(a_s^*)_{{{}}}$'.format(h_idx), fontsize=fontsize, rotation=0, labelpad=30, va='center')
        axes[i].set_ylim(None, max_anom + 0.05 * max_anom)
        axes[i].hlines(joint_threshold[j], xmin=0, xmax=len(a_s_star[j]) - 1, linestyle='--', color='black')

    # compute the allowed anomaly index range
    true_anom_ind = []
    for j, y in enumerate(Y):
        if y == 1:
            true_anom_ind.append(j)
    allowed_anom_ind_rng = (true_anom_ind[0] - 100, true_anom_ind[-1] + 100)

    # plot: compute top-k accuracies
    local_maxima_ind, _ = find_peaks(a_final, distance=100)
    local_maxima = a_final[local_maxima_ind]
    local_maxima = {maxima: maxima_idx for maxima, maxima_idx in zip(local_maxima, local_maxima_ind)}
    local_maxima = dict(sorted(local_maxima.items(), key=lambda item: item[0]))
    top1_idx = [np.argmax(a_final, axis=0)]
    topk_ind = {'top-1': top1_idx,
                'top-3': list(local_maxima.values())[-3:],
                'top-5': list(local_maxima.values())[-5:]}
    topk_acc = {k: None for k in topk_ind.keys()}
    for k, v in topk_ind.items():
        accs = []
        for pred_idx in v:
            acc = 1. if (allowed_anom_ind_rng[0] <= pred_idx <= allowed_anom_ind_rng[-1]) else 0.
            accs.append(acc)
        topk_acc[k] = 1 if np.mean(accs) > 0 else 0

    # plot: bar{a}_s^*
    i += 1
    axes[i].plot(a_bar_s_star, color='darkturquoise')
    axes[i].set_xticks([])
    axes[i].set_xlim(0, len(a_bar_s_star) - 1)
    axes[i].set_ylabel(r'$\bar{a}_s^*$', fontsize=fontsize, rotation=0, labelpad=15, va='center')
    axes[i].hlines(final_threshold, xmin=0, xmax=len(a_bar_s_star) - 1, linestyle='--', color='black')

    # plot: doublebar{a}_s^*
    i += 1
    rng = np.arange(len(a_2bar_s_star))
    axes[i].plot(rng, a_2bar_s_star, color='royalblue')
    axes[i].set_xticks([])
    axes[i].set_xlim(0, len(a_2bar_s_star) - 1)
    axes[i].set_ylabel(r'$\bar{\bar{a}}_s^*$', fontsize=fontsize, rotation=0, labelpad=15, va='center')
    axes[i].hlines(final_threshold, xmin=0, xmax=len(a_2bar_s_star) - 1, linestyle='--', color='black')

    # plot: a_final
    i += 1
    rng = np.arange(len(a_final))
    axes[i].plot(rng, a_final, color='red', label=f'acc: {topk_acc}')
    axes[i].scatter(topk_ind['top-5'], a_final[topk_ind['top-5']], marker='*', alpha=1.0, color='cyan')
    axes[i].scatter(topk_ind['top-1'], a_final[topk_ind['top-1']], marker='^', alpha=1.0, color='blue')
    axes[i].set_xticks([])
    axes[i].set_xlim(0, len(a_final) - 1)
    axes[i].set_ylabel(r'$a_{final}$', fontsize=fontsize, rotation=0, labelpad=25, va='center')
    axes[i].legend(loc='upper right')
    axes[i].hlines(final_threshold, xmin=0, xmax=len(a_final) - 1, linestyle='--', color='black')

    # plot: explainable sampling
    if args.explainable_sampling:
        # load model
        input_length = window_size = set_window_size(dataset_idx, config['dataset']['n_periods'])
        stage2 = ExpStage2.load_from_checkpoint(os.path.join('saved_models', f'stage2-{dataset_idx}.ckpt'), 
                                                dataset_idx=dataset_idx, input_length=input_length, config=config, 
                                                map_location=f'cuda:{args.device}')
        maskgit = stage2.maskgit
        maskgit.eval()

        i += 1
        for timestep_idx, timestep in enumerate(timestep_rng_test):
            print(f'explainable sampling.. {round(timestep_idx / len(timestep_rng_test) * 100)}%')
            # fetch a window at each timestep
            window_rng = slice(timestep, timestep + window_size)
            x_unscaled = X_test_unscaled[window_rng]  # (window_size,)
            mu = np.nanmean(x_unscaled, axis=-1, keepdims=True)  # (1,)
            sigma = np.nanstd(x_unscaled, axis=-1, keepdims=True)  # (1,)
            min_std = 1.e-4  # to prevent numerical instability in scaling.
            sigma = np.clip(sigma, min_std, None)
            x = (x_unscaled - mu) / sigma  # (window_size,)

            # encode
            z_q, s = maskgit.encode_to_z_q(torch.from_numpy(x[None, None, :]).to(args.device), maskgit.encoder,
                                           maskgit.vq_model)  # s: (1 n)
            latent_height, latent_width = z_q.shape[2], z_q.shape[3]

            # anomaly locations
            anom_window = a_final[window_rng]  # (ts_len',)
            anom_window = torch.from_numpy(anom_window)[None, None, :]  # (1 1 ts_len')
            anom_window = torch.nn.functional.interpolate(anom_window, size=(latent_width,))[0, 0]  # (latent_width,)
            is_anom = anom_window > final_threshold  # (latent_width,)
            if is_anom.float().mean().item() > args.max_masking_rate_for_explainable_sampling:  # if masked more than n%, we leave 10% to give a minimum context
                tau = torch.quantile(anom_window, 1 - args.max_masking_rate_for_explainable_sampling).item()
                is_anom = anom_window > tau

            if is_anom.float().mean().item() > 0.:
                # sample
                s_star = rearrange(s, '1 (h w) -> 1 h w', h=latent_height)  # (1, latent_height, latent_width)
                s_star[:, :, is_anom] = maskgit.mask_token_id
                s_star = rearrange(s_star, '1 h w -> 1 (h w)')  # (1 n)
                s_star = repeat(s_star, '1 n -> b n', b=args.n_explainable_samples)  # b: user parameter
                masking_ratio = is_anom.int().sum() / len(is_anom)
                t_star = int(np.floor(2 * np.arccos(masking_ratio) / np.pi * config['MaskGIT']['T']))  # const.
                s_Tstar = maskgit.explainable_sampling(t_star, s_star)  # (b n)
                xhat = maskgit.decode_token_ind_to_timeseries(s_Tstar).cpu().numpy()  # (b, n_channels, window_size)
                xhat = (xhat * sigma) + mu  # unscaled; (b, n_channels, window_size)
                xhat = xhat[:, 0, :]  # univariate; (b, window_size,)

                # plot
                for b in range(xhat.shape[0]):
                    axes[i].plot(np.arange(timestep, timestep + window_size), xhat[b], alpha=0.5, color='C1')
            else:
                axes[i].plot(np.arange(timestep, timestep + window_size), x_unscaled, alpha=0.5, color='black')
        axes[i].set_xticks([])
        axes[i].set_xlim(0, len(X_test_unscaled) - 1)
        axes[i].set_ylabel(r'counter\nfactual', fontsize=fontsize)

    # save: fig
    plt.tight_layout()
    topk_acc_vals = ','.join([str(round(v)) for v in list(topk_acc.values())])
    plt.savefig(get_root_dir().joinpath('evaluation', 'results', f'{dataset_idx}-joint_anomaly_score-acc_{topk_acc_vals}.png'))
    plt.close()
