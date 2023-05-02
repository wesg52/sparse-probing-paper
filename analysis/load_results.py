import os
import json
import pickle
import pandas as pd
import numpy as np


def load_probing_experiment_results(results_dir, experiment_name, model_name, dataset_name, inner_loop, uncollapse_features=False):
    result_dir = os.path.join(
        results_dir, experiment_name, model_name, dataset_name, inner_loop)
    results = {}
    config = None
    for result_file in os.listdir(result_dir):
        if result_file == 'config.json':
            config_file = os.path.join(result_dir, result_file)
            config = json.load(open(config_file, 'r'))
            continue
        # example: heuristic_sparsity_sweep.arxiv.pythia-125m.mlp,hook_post.max.0.p
        _, feature, _, hook_loc, aggregation, layer, _ = result_file.split('.')
        layer = int(layer)
        hook_loc = hook_loc.replace(',', '.')
        results_dict = pickle.load(
            open(os.path.join(result_dir, result_file), 'rb'))
        if uncollapse_features:  # --save_features_together enabled
            for k, v in results_dict.items():
                results[(f'{k}', layer, aggregation, hook_loc)] = v
        else:
            results[feature, layer, aggregation, hook_loc] = results_dict
    return results, config


def load_probing_experiment_results_old(results_dir, experiment_name, inner_loop, model_name):
    # old version
    result_dir = os.path.join(
        results_dir, experiment_name, inner_loop, model_name)
    results = {}
    for result_file in os.listdir(result_dir):
        if len(result_file.split('.')) == 5:
            _, feature, _, layer, file_type = result_file.split('.')
        else:
            continue
            print(result_file)
            _, feature, probe_loc,  _, layer, file_type = result_file.split(
                '.')
        layer = int(layer[1:])
        if feature not in results:
            results[feature] = {}
        results[feature][layer] = pickle.load(
            open(os.path.join(result_dir, result_file), 'rb'))
    return results


def make_heuristic_probing_results_df(results_dict):
    flattened_results = {}
    for feature in results_dict:
        for layer in results_dict[feature]:
            for sparsity in results_dict[feature][layer]:
                flattened_results[(feature, layer, sparsity)
                                  ] = results_dict[feature][layer][sparsity]
    rdf = pd.DataFrame(flattened_results).T.sort_index().rename_axis(
        index=['feature', 'layer', 'k'])
    return rdf


def collect_monosemantic_results(probing_results):
    dfs = {}
    for k, result in probing_results.items():
        dfs[k] = pd.DataFrame(result).T
    rdf = pd.concat(dfs)  # .reset_index()
    rdf.rename_axis(
        index=['feature', 'layer', 'aggregation', 'hook_loc', 'neuron'],
        inplace=True
    )
    return rdf.sort_index()
