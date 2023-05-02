import datetime
import random
import numpy as np
import torch
import argparse
import time


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def timestamp():
    return datetime.datetime.now().strftime("%Y:%m:%d:%H:%M:%S")


def default_argument_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # experiment params
    parser.add_argument(
        '--experiment_name', default=str(int(time.time()) // 10),
        help='Name of experiment to save')
    parser.add_argument(
        '--experiment_type', nargs='+', required=True,
        help='The inner loop function(s) to run for the experiment')
    parser.add_argument(
        '--model', default='pythia-70m',
        help='Name of model from TransformerLens')
    parser.add_argument(
        '--feature_dataset',
        help='Name of cached feature dataset')
    parser.add_argument(
        '--probe_location', default='mlp.hook_post',
        help='Model component to probe')
    parser.add_argument(
        '--activation_aggregation', default=None,
        help='Average activations across all tokens in a sequence')
    parser.add_argument(
        '--seed', default=1, type=int,
        help='Random seed for experiment')
    parser.add_argument(
        '--probe_next_token_feature', action='store_true',
        help='Probe the token before the probe_index to predict property of the probe_index')
    return parser


MODEL_N_LAYERS = {
    'pythia-70m': 6,
    'pythia-160m': 12,
    'pythia-410m': 24,
    'pythia-1b': 16,
    'pythia-1.4b': 24,
    'pythia-2.8b': 32,
    'pythia-6.9b': 32
}
