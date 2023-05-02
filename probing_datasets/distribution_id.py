import os
import datasets
import torch
import numpy as np
from .common import *

DATASET_SPLITS = {
    'Wikipedia (en)': 'wikipedia',
    'PubMed Abstracts': 'pubmed_abstracts',
    'StackExchange': 'stack_exchange',
    'Github': 'github',
    'ArXiv': 'arxiv',
    'USPTO Backgrounds': 'uspto',
    'FreeLaw': 'freelaw',
    'HackerNews': 'hackernews',
    'Enron Emails': 'enron'
}


class DataDistributionIDFeatureDataset(FeatureDataset):

    def __init__(self, name, data_splits):
        self.name = name
        self.data_splits = data_splits

    def prepare_dataset(self, exp_cfg):
        """Convert categorial labels into binary labels for each data_splits.

        Returns: tokenized_dataset, feature_datasets with structure:
            {data_splits: (indices, classes)}.

        ...except when exp_cfg.aggregation is not None. Then
        indices are valid_index mask with class per row."""
        dataset = self.load(exp_cfg.dataset_cfg)
        _, n = dataset['probe_indices'].shape

        feature_datasets = {k: {'indices': [], 'classes': []}
                            for k in self.data_splits.values()}

        dataset_distribution = np.array(dataset['distribution'])

        if exp_cfg.activation_aggregation is None:
            valid_index_mask = torch.zeros_like(dataset['tokens'])
            for ix, valid_seq_indices in enumerate(dataset['probe_indices']):
                valid_index_mask[ix, valid_seq_indices] = 1
            valid_indices = np.where(valid_index_mask.flatten())[0]

            extended_dataset_distribution = dataset_distribution[:, None].repeat(
                n, axis=1).flatten()
            for ix, split in enumerate(self.data_splits.values()):
                split_label = (extended_dataset_distribution ==
                               split).astype(int) * 2 - 1
                # same index_mask for all splits
                feature_datasets[split] = (valid_indices, split_label)
            return dataset, feature_datasets

        else:  # each feature dataset is (size(n x ctx_len) index_mask, size(n) class)
            valid_index_mask = torch.zeros_like(dataset['tokens'])
            for ix, valid_seq_indices in enumerate(dataset['valid_indices']):
                valid_index_mask[ix, valid_seq_indices] = 1

            for ix, split in enumerate(self.data_splits.values()):
                split_label = (dataset_distribution ==
                               split).astype(int) * 2 - 1
                # same index_mask for all languages
                feature_datasets[split] = (valid_index_mask, split_label)
            return dataset, feature_datasets

    def make(self, dataset_config, args, raw_dataset, tokenizer, cache=True):

        n_probe_tokens = args.get('lang_id_n_tokens', 2)
        ignore_k = args.get('ignore_first_k', 25)

        tokenized_dataset = tokenize_and_concatenate_separate_subsequences(
            raw_dataset,
            tokenizer,
            max_length=dataset_config.ctx_len,
            add_bos_token=args.get('add_bos_token', True),
        )
        distribution_ids = [self.data_splits[meta['pile_set_name']]
                            for meta in raw_dataset['meta']]
        feature_dataset = tokenized_dataset.add_column(
            'distribution', distribution_ids)
        # filter out too short sequences.
        feature_dataset = feature_dataset.filter(
            lambda x: len(x['all_tokens']) > ignore_k + n_probe_tokens + 1)

        all_probe_indices = []
        all_valid_indices = []

        END_TOKENS = torch.tensor(
            [tokenizer.eos_token_id, tokenizer.pad_token_id], dtype=torch.long)
        for i in range(len(feature_dataset)):
            # determine last valid token.
            if feature_dataset[i]['tokens'][-1] not in END_TOKENS:
                eos = len(feature_dataset[i]['tokens']) - 1
            else:
                eos = torch.isin(
                    feature_dataset[i]['tokens'],
                    END_TOKENS
                ).nonzero()[0].item()

            valid_indices = np.arange(ignore_k, eos)
            if len(valid_indices) > n_probe_tokens:
                probe_indices = sorted(np.random.choice(
                    valid_indices, size=n_probe_tokens, replace=False).tolist())
            else:
                probe_indices = valid_indices.tolist()

            all_probe_indices.append(probe_indices)
            all_valid_indices.append(valid_indices)

        # valid indices are required for aggregation.
        feature_dataset = feature_dataset.add_column(
            'probe_indices', all_probe_indices)
        feature_dataset = feature_dataset.add_column(
            'valid_indices', all_valid_indices)

        feature_dataset = feature_dataset.filter(
            lambda x: len(x['valid_indices']) > 0)

        feature_dataset.set_format('torch')

        if cache:
            self.save(dataset_config, feature_dataset)

        return feature_dataset


DISPLAY_NAMES = {
    'wikipedia': 'Wikipedia',
    'pubmed_abstracts': 'PubMed',
    'stack_exchange': 'StackEx',
    'github': 'Github',
    'arxiv': 'ArXiv',
    'uspto': 'USPTO',
    'freelaw': 'FreeLaw',
    'hackernews': 'HackNews',
    'enron': 'Enron'
}
