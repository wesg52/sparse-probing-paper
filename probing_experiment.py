import os
import pickle
import random
import time
import argparse
import math
import json

import torch
import numpy as np
from sklearn.metrics import *

from load import *
from config import *
from utils import timestamp, seed_all, default_argument_parser, MODEL_N_LAYERS
from make_feature_datasets import prepare_feature_dataset
from activations.activation_probing_dataset import make_index_mask, load_activation_probing_dataset

from experiments.activations import *
from experiments.probes import *
from experiments.metrics import *
from experiments.inner_loops import *


def save_result(exp_cfg, result, inner_loop_name, feature_name):
    save_path = os.path.join(
        os.getenv('RESULTS_DIR', 'results'),
        exp_cfg.experiment_name,
        exp_cfg.model_name,
        exp_cfg.feature_dataset,
        inner_loop_name
    )
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
        config_dict = vars(exp_cfg)
        if 'dataset_cfg' in config_dict:
            del config_dict['dataset_cfg']
        with open(os.path.join(save_path, 'config.json'), 'w') as f:
            json.dump(config_dict, f)

    model_name = exp_cfg.model_name.replace('.', ',')
    probe_location = exp_cfg.probe_location.replace('.', ',')
    aggregation = exp_cfg.activation_aggregation
    aggregation = aggregation if aggregation is not None else "none"

    save_name = f'{inner_loop_name}.{feature_name}.{model_name}.{probe_location}.{aggregation}.{layer}.p'
    save_file = os.path.join(save_path, save_name)
    pickle.dump(result, open(save_file, 'wb'))


def get_skip_features(exp_cfg, feature_names):
    feature_collection = exp_cfg.feature_dataset.split('.')[0]
    skip_features = set()
    if feature_collection == 'ewt':
        if exp_cfg.feature_subset == 'upos':
            skip_features = set([
                f for f in feature_names if not f.startswith('upos')
            ])
        elif exp_cfg.feature_subset == 'dep':
            skip_features = set([
                f for f in feature_names if not f.startswith('dep')
            ])
        elif exp_cfg.feature_subset == 'morph':
            skip_features = set([
                f for f in feature_names if f.startswith('upos') or f.startswith('dep')
            ])
        elif exp_cfg.feature_subset == 'not-dep':
            skip_features = set([
                f for f in feature_names if f.startswith('dep')
            ])
    elif feature_collection == 'compound_words':
        if exp_cfg.feature_subset:
            features_to_keep = set(exp_cfg.feature_subset.split(','))
            skip_features = set([
                k for k in feature_datasets.keys() if k not in features_to_keep
            ])
    return skip_features


def run_probe_on_layer(exp_cfg, tokenized_dataset, feature_datasets, layer):
    # TODO: add option to compute activations on the fly
    activation_dataset = load_activation_probing_dataset(exp_cfg, layer)
    index_mask = make_index_mask(exp_cfg, tokenized_dataset, feature_datasets)

    print(f'{timestamp()} finished loading activations for layer {layer}')

    # Run the probing experiments
    skip_features = get_skip_features(exp_cfg, feature_datasets.keys())
    results = {inner_loop: {} for inner_loop in exp_cfg.experiment_type}
    for feature_name, feature_data in feature_datasets.items():
        # prepare the feature specific dataset
        if feature_name in skip_features:
            continue
        if feature_data == None:
            print(f'Warning no feature data for {feature_name}')
            continue

        if exp_cfg.activation_aggregation is None:
            # filter the activation dataset to only include the indices required
            feature_indices, feature_classes = feature_data
            all_required_indices = np.where(index_mask.flatten())[0]
            feature_index_mask = np.isin(all_required_indices, feature_indices)
            feature_activation_dataset = activation_dataset[feature_index_mask, :]
        else:
            feature_ix_mask, feature_classes = feature_data
            all_required_seqs = np.where(feature_ix_mask.sum(axis=1) > 0)[0]
            feature_activation_dataset = activation_dataset[all_required_seqs, :]

        # run each of the inner loops on the feature specific dataset
        for inner_loop_name in exp_cfg.experiment_type:
            inner_loop_fn = INNER_LOOP_FNS.get(inner_loop_name, None)
            if inner_loop_fn is None:
                raise ValueError(
                    f'{inner_loop_name} is not a valid experiment type')

            result = inner_loop_fn(
                exp_cfg,
                feature_activation_dataset.astype(np.float32),
                feature_classes
            )
            print(
                f'{timestamp()} | {feature_name} | {inner_loop_name} | {exp_cfg.model_name} | {layer}')

            if exp_cfg.save_features_together:
                results[inner_loop_name][feature_name] = result
            else:
                save_result(exp_cfg, result, inner_loop_name, feature_name)

    if exp_cfg.save_features_together:
        for inner_loop_name in exp_cfg.experiment_type:
            save_result(
                exp_cfg, results[inner_loop_name], inner_loop_name, 'all')


if __name__ == "__main__":
    # see utils.py for args
    parser = default_argument_parser()
    # experiment params
    parser.add_argument(
        '--normalize_activations', action='store_true',
        help='Normalize activations per neuron to have standard deviation 0.1')
    parser.add_argument(
        '--test_set_frac', default=0.3, type=float,
        help='Fraction of dataset to use as test set')
    parser.add_argument(
        '--save_features_together', action='store_true',
        help='Save features together in a single file')
    parser.add_argument(
        '--feature_subset', default='', type=str,
        help='Subset of features to use (functionality determined by feature dataset prepare_dataset())')
    # probe params
    parser.add_argument(
        '--heuristic_feature_selection_method', default='mean_dif', type=str,
        help='Method feature selection (eg, mean_dif, f_stat, mi')
    parser.add_argument(
        '--osp_heuristic_filter_size', default=50, type=int,
        help='Size of initial heuristic feature selection before osp')
    parser.add_argument(
        '--max_k', default=256, type=int,
        help='Max k to use for any inner loop')
    parser.add_argument(
        '--osp_upto_k', default=8, type=int, help='Highest k to use optimal sparse probing')
    parser.add_argument(
        '--gurobi_timeout', default=60, type=int, help='Max time (seconds) to let Gurobi solve')
    parser.add_argument(
        '--gurobi_verbose', default=False, type=bool, help='Print out full gurobi logs')
    # inner loop specific params
    parser.add_argument(
        '--iterative_pruning_fixed_k', default=5, type=int,
        help='Value of k to hold fixed while implementing iterative pruning')
    parser.add_argument(
        '--iterative_pruning_n_prune_steps', default=10, type=int,
        help='Number of steps to take in iterative pruning')

    args = vars(parser.parse_args())
    feature_dataset_cfg = parse_dataset_args(args['feature_dataset'])
    exp_cfg = ExperimentConfig(args, feature_dataset_cfg)

    seed_all(exp_cfg.seed)

    feature_dataset_info = prepare_feature_dataset(exp_cfg)
    tokenized_dataset, feature_datasets = feature_dataset_info

    print(f'{timestamp()} finished preparing dataset')

    n_layers = MODEL_N_LAYERS[args['model']]
    # for parallelization on cluster
    task_id = int(os.getenv('SLURM_ARRAY_TASK_ID', 1))
    task_count = int(os.getenv('SLURM_ARRAY_TASK_COUNT', 1))
    layers = list(range(task_id-1, n_layers, task_count))

    for layer in layers:
        run_probe_on_layer(
            exp_cfg,
            tokenized_dataset,
            feature_datasets,
            layer
        )
