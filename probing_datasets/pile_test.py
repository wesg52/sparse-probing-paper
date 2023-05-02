import os
import datasets
import torch
import numpy as np
from .common import *
from transformer_lens.utils import tokenize_and_concatenate


class PileTestSplitFeatureDataset(FeatureDataset):
    """
    A large tokenized text dataset corresponding to the Pile test set.

    Tokenized to have minimal padding but where all concatenated sequences
    come from the same sub distribution (with the label).
    """

    def __init__(self, name):
        self.name = name

    def prepare_dataset(self, exp_cfg):
        raise NotImplementedError(
            "PileTestSplitFeatureDataset is not meant to be used for probing, only activation statistics."
        )

    def make(self, dataset_config, args, raw_dataset, tokenizer, cache=True):

        data_split = np.array([m['pile_set_name']
                              for m in raw_dataset['meta']])
        splits = np.unique(data_split)

        sub_datasets = []
        for split in splits:
            print(f"Tokenizing {split}...")
            sub_dataset_indices = np.where(data_split == split)[0]
            sub_dataset = raw_dataset.select(sub_dataset_indices)

            tokenized_sub_dataset = tokenize_and_concatenate(
                sub_dataset,
                tokenizer,
                max_length=dataset_config.ctx_len,
                add_bos_token=args.get('add_bos_token', True),
            )
            tokenized_sub_dataset = tokenized_sub_dataset.add_column(
                'distribution',
                [split for _ in range(len(tokenized_sub_dataset))]
            )
            sub_datasets.append(tokenized_sub_dataset)

        tokenized_dataset = datasets.concatenate_datasets(sub_datasets)

        if cache:
            self.save(dataset_config, tokenized_dataset)

        return tokenized_dataset
