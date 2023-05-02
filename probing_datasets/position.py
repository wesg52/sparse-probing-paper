import os
import datasets
import einops
import torch
import numpy as np
from .common import *
from .pile_test import PileTestSplitFeatureDataset
from transformer_lens.utils import tokenize_and_concatenate


class PositionFeatureDataset(FeatureDataset):
    """
    Pile test set with minimal padding and position columns.
    """

    def __init__(self):
        self.name = 'position'

    def prepare_dataset(self, exp_cfg):
        dataset = self.load(exp_cfg.dataset_cfg)
        index_mask = dataset['index_mask']
        probe_indices = np.where(index_mask.flatten())[0]

        pos_cols = ['abs_pos', 'norm_abs_pos',
                    'rel_pos', 'norm_rel_pos', 'log_pos']
        feature_datasets = {
            col: (probe_indices, dataset[col][index_mask].numpy())
            for col in pos_cols
        }
        return dataset, feature_datasets

    def make(self, dataset_config, args, raw_dataset, tokenizer, cache=True):

        # reuse PileTestSplitFeatureDataset make()
        tokenized_dataset = PileTestSplitFeatureDataset('').make(
            dataset_config, args, raw_dataset, tokenizer, cache=False)
        tokenized_dataset = tokenized_dataset.select(
            range(args.get('n_seqs', 10_000)))

        n, d, = tokenized_dataset['tokens'].shape

        abs_pos = np.arange(d).astype(np.float32)
        norm_abs_pos = abs_pos / abs_pos.std()

        rel_pos = abs_pos - abs_pos.mean()
        norm_rel_pos = rel_pos / rel_pos.std()

        log_pos = np.log2(abs_pos + 1)

        # add position columns to dataset
        tokenized_dataset = tokenized_dataset.add_column(
            'abs_pos', [abs_pos for _ in range(n)])
        tokenized_dataset = tokenized_dataset.add_column(
            'norm_abs_pos', [norm_abs_pos for _ in range(n)])
        tokenized_dataset = tokenized_dataset.add_column(
            'rel_pos', [rel_pos for _ in range(n)])
        tokenized_dataset = tokenized_dataset.add_column(
            'norm_rel_pos', [norm_rel_pos for _ in range(n)])
        tokenized_dataset = tokenized_dataset.add_column(
            'log_pos', [log_pos for _ in range(n)])

        s = args.get('dataset_size', 50_000)
        probe_indices = np.random.choice(n*d, s, replace=False)
        index_mask = np.zeros(n*d, dtype=bool)
        index_mask[probe_indices] = True
        index_mask = index_mask.reshape((n, d))

        tokenized_dataset = tokenized_dataset.add_column(
            'index_mask', [index_mask[i] for i in range(n)])

        if cache:
            self.save(dataset_config, tokenized_dataset)

        return tokenized_dataset
