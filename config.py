import os
from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    def __init__(self, args, feature_dataset_cfg):
        self.experiment_name = args.get('experiment_name')
        self.experiment_type = args.get('experiment_type')
        self.model_name = args.get('model')
        self.dataset_cfg = feature_dataset_cfg
        self.feature_dataset = args['feature_dataset']
        self.probe_location = args.get('probe_location', 'mlp.hook_post')
        self.activation_aggregation = args.get('activation_aggregation', None)
        self.normalize_activations = args.get('normalize_activations', False)
        self.seed = args.get('seed', 1)
        self.test_set_frac = args.get('test_set_frac', 0.3)
        self.batch_size = args.get('batch_size', 16)
        self.save_features_together = args.get(
            'save_features_together', False)
        self.feature_subset = args.get('feature_subset', '')
        self.probe_next_token_feature = args.get(
            'probe_next_token_feature', False)
        self.heuristic_feature_selection_method = args.get(
            'heuristic_feature_selection_method', 'mean_dif')
        self.max_k = args.get('max_k', 128)
        self.osp_upto_k = args.get('osp_upto_k', 5)
        self.osp_heuristic_filter_size = args.get(
            'osp_heuristic_filter_size', 50)
        self.gurobi_timeout = args.get('gurobi_timeout', 60)
        self.gurobi_verbose = args.get('gurobi_verbose', False)
        self.iterative_pruning_fixed_k = args.get(
            'iterative_pruning_fixed_k', 5)
        self.iterative_pruning_n_prune_steps = args.get(
            'iterative_pruning_n_prune_steps', 10)
        self.max_per_class = args.get('max_per_class', -1)


@dataclass
class FeatureDatasetConfig:
    def __init__(
        self,
        dataset_name,
        tokenizer_name,
        ctx_len,
        n_sequences,
    ):
        self.dataset_name = dataset_name
        self.tokenizer_name = tokenizer_name
        self.ctx_len = ctx_len
        self.n_sequences = n_sequences

    def make_dir_name(self):
        save_dir = '.'.join([
            self.dataset_name,
            self.tokenizer_name,
            str(self.ctx_len),
            str(self.n_sequences),
        ])
        return save_dir


def parse_dataset_args(feature_dataset_string):
    ds_args = feature_dataset_string.split('.')
    feature_collection, tokenizer_name, seq_len, n_seqs = ds_args
    feature_dataset_cfg = FeatureDatasetConfig(
        feature_collection,
        tokenizer_name,
        int(seq_len),
        int(n_seqs),
    )
    return feature_dataset_cfg
