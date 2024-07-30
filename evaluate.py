from argparse import ArgumentParser
import pickle
from multiprocessing import Process

import numpy as np

from utils import get_root_dir, load_yaml_param_settings, set_window_size
from evaluation import evaluate_fn, save_final_summarized_figure


def load_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help="Path to the config data  file.", default=get_root_dir().joinpath('configs', 'config.yaml'))
    parser.add_argument('--dataset_ind', default=[1,], help='e.g., 1 2 3. Indices of datasets to run experiments on.', nargs='+', required=True)
    parser.add_argument('--latent_window_size_rates', default=[0.1, 0.3, 0.5], nargs='+')
    parser.add_argument('--rolling_window_stride_rate', default=0.1, type=float, help='stride = rolling_window_stride_rate * window_size')
    parser.add_argument('--q', default=0.99, type=float)
    parser.add_argument('--explainable_sampling', default=False, help='Note that this script will run more slowly with this option being True.')
    parser.add_argument('--n_explainable_samples', type=int, default=2, help='how many explainable samples to get per window.')
    parser.add_argument('--max_masking_rate_for_explainable_sampling', type=float, default=0.9, help='it prevents complete masking and ensures the minimum valid tokens to leave a minimum context for explainable sampling.')
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--n_workers', default=4, type=int, help='multi-processing for latent_window_size_rate.')
    return parser.parse_args()


def process_list_arg(arg, dtype):
    arg = np.array(arg, dtype=dtype)
    return arg

def process_bool_arg(arg):
    if str(arg) == 'True':
        arg = True
    elif str(arg) == 'False':
        arg = False
    else:
        raise ValueError
    return arg


if __name__ == '__main__':
    # load config
    args = load_args()
    config = load_yaml_param_settings(args.config)

    args.dataset_ind = process_list_arg(args.dataset_ind, int)
    args.latent_window_size_rates = process_list_arg(args.latent_window_size_rates, float)
    args.explainable_sampling = process_bool_arg(args.explainable_sampling)
    for idx in args.dataset_ind:
        print(f'\nidx: {idx}')
        idx = int(idx)

        for worker_idx in range(len(args.latent_window_size_rates)):
            latent_window_size_rates = args.latent_window_size_rates[worker_idx*args.n_workers: (worker_idx+1)*args.n_workers]
            if len(latent_window_size_rates) == 0:
                break

            procs = []
            for wsr in latent_window_size_rates:
                proc = Process(target=evaluate_fn, args=(config, idx, wsr, args.rolling_window_stride_rate, args.q, args.device))  # make sure to put , (comma) at the end
                procs.append(proc)
                proc.start()
            for p in procs:
                p.join()  # make each process wait until all the other process ends.

        # integrate all the joint anomaly scores across `latent_window_size_rates`
        a_s_star = 0.
        joint_threshold = 0.
        for wsr in args.latent_window_size_rates:
            result_fname = get_root_dir().joinpath('evaluation', 'results', f'{idx}-anomaly_score-latent_window_size_rate_{wsr}.pkl')
            with open(str(result_fname), 'rb') as f:
                result = pickle.load(f)
                a_star = result['a_star']  # (n_freq, ts_len')
                a_s_star += a_star  # (n_freq, ts_len')
                joint_threshold += result['anom_threshold']

        # \bar{a}_s^star
        a_bar_s_star = a_s_star.mean(axis=0)  # (ts_len',)

        # \doublebar{a}_s^star
        window_size = set_window_size(idx, config['dataset']['n_periods'])
        a_2bar_s_star = np.zeros_like(a_bar_s_star)  # (ts_len',)
        for j in range(len(a_2bar_s_star)):
            rng = slice(max(0, j - window_size // 2), j + window_size // 2)
            a_2bar_s_star[j] = np.mean(a_bar_s_star[rng])

        # a_final
        a_final = (a_bar_s_star + a_2bar_s_star) / 2

        # final threshold
        final_threshold = joint_threshold.mean()
        anom_ind = a_final > final_threshold

        # plot
        save_final_summarized_figure(idx, result['X_test_unscaled'], result['Y'], result['timestep_rng_test'],
                                     a_s_star, a_bar_s_star, a_2bar_s_star, a_final,
                                     joint_threshold, final_threshold, anom_ind, window_size, config, args)

        # save: resulting data
        joint_resulting_data = {'dataset_index': idx,
                                'X_test_unscaled': result['X_test_unscaled'],  # time series
                                'Y': result['Y'],  # label

                                'a_s^*': a_s_star,  # (n_freq, ts_len')
                                'bar{a}_s^*': a_bar_s_star,  # (ts_len',)
                                'doublebar{a}_s^*': a_2bar_s_star,  # (ts_len',)
                                'a_final': a_final,  # (ts_len',)

                                'joint_threshold': joint_threshold,  # (n_freq,)
                                'final_threshold': final_threshold  # (,)
                                }
        saving_fname = get_root_dir().joinpath('evaluation', 'results', f'{idx}-joint_anomaly_score.pkl')
        with open(saving_fname, 'wb') as f:
            pickle.dump(joint_resulting_data, f, pickle.HIGHEST_PROTOCOL)
