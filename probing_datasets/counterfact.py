import sys

import torch
import numpy as np
from datasets.arrow_dataset import Dataset
from transformers import AutoTokenizer

from probing_datasets.common import FeatureDataset
from config import FeatureDatasetConfig


class CounterfactFeatureDataset(FeatureDataset):

    def __init__(self):
        pass

    def prepare_dataset(self, exp_cfg):
        '''
        Return valid indices and classes for the feature dataset.
        '''
        dataset = self.load(exp_cfg.dataset_cfg)

        # index is position within flattened (n_seq x seq_len,) array
        feature_indices = dataset['target_end'] - 1
        feature_indices += torch.arange(len(feature_indices)) * len(dataset[0]['tokens'])

        # classes to {-1, +1}
        feature_classes = np.full(len(dataset), -1)
        feature_classes[dataset['text_true']] = 1

        feature_datasets = {'text_true': (feature_indices, feature_classes)}
        return dataset, feature_datasets

    def make(
        self,
        dataset_config: FeatureDatasetConfig,
        args: dict,  # command line arguments from make_feature_datasets
        raw_dataset: Dataset,
        tokenizer: AutoTokenizer,
        cache=True,
    ) -> Dataset:
        '''
        Returns feature_dataset with columns:
            text: raw strings
            tokens: tokenized strings of consistent length
            text_true: boolean class label
            subject_start: index of the start of the subject
            subject_end: index of the end of the subject
            target_start: index of the start of the target
            target_end: index of the end of the target
            relation_id: relation id
        '''

        # create a positive and negative example for each in the batch
        def create_pos_neg(batch):
            text = [(p + tt, p + tf) for p, tt, tf in zip(batch['prompt'], batch['target_true'], batch['target_false'])]
            text_flat = [t for pair in text for t in pair]
            return {
                'text': text_flat,
                'text_true': len(batch['prompt']) * [True, False],
                'relation_prefix': [r for r in batch['relation_prefix'] for _ in range(2)],
                'subject': [s for s in batch['subject'] for _ in range(2)],
                'prompt': [p for p in batch['prompt'] for _ in range(2)],
                'relation_id': [rid for rid in batch['relation_id'] for _ in range(2)],
            }
        remove_columns = ['relation', 'relation_suffix', 'target_true_id', 'target_false_id',
                          'target_true', 'target_false', 'subject']
        feature_dataset = raw_dataset.map(create_pos_neg, batched=True, remove_columns=remove_columns)

        # tokenize each example, storing the indices for the subject and target
        def tokenize(example):
            # get tokens
            seq_len = dataset_config.ctx_len - 1 if args['add_bos'] else dataset_config.ctx_len
            all_tokens = tokenizer(example['text'], max_length=seq_len, truncation=True, padding='max_length')['input_ids']
            tokens = [tokenizer.bos_token_id] + all_tokens if args['add_bos'] else all_tokens

            # get the index of the subject and target
            # NOTE: this isn't the most efficient but it doesn't need to be and should be pretty robust
            token_strs = tokenizer.batch_decode(tokens, clean_up_tokenization_spaces=True)
            sentence = ""
            index_counter = 0
            search_str = tokenizer.bos_token + example['relation_prefix'] if args['add_bos'] else example['relation_prefix']
            for i, token_str in enumerate(token_strs + ['']):
                if index_counter == 0:  # start of subject
                    if search_str in sentence:
                        example['subject_start'] = i
                        search_str += example['subject']
                        index_counter += 1
                if index_counter == 1:  # (1 after) end of subject
                    if search_str in sentence:
                        example['subject_end'] = i
                        search_str = example['prompt']
                        index_counter += 1
                if index_counter == 2:  # start of target
                    if search_str in sentence:
                        example['target_start'] = i
                        search_str = example['text']
                        index_counter += 1
                if index_counter == 3:  # (1 after) end of target
                    if search_str in sentence:
                        example['target_end'] = i
                        break
                sentence += token_str

            # print(example['text'], token_strs[probe_indices[0]:probe_indices[1]], token_strs[probe_indices[2]:probe_indices[3]], probe_indices)

            example['tokens'] = tokens
            # valid_indicies_min = 1 if args['add_bos'] else 0
            # valid_indicies_max = min(len(tokenizer(example['text'])['input_ids']) + valid_indicies_min, dataset_config.ctx_len)
            # example['valid_indices'] = list(range(valid_indicies_min, valid_indicies_max))

            return example

        feature_dataset = feature_dataset.map(tokenize)

        # this is a bit messy but sometimes we have a problem with the tokenization decoding (e.g. uses weird characters) and we just skip the example
        probe_indices_keys = ['subject_start', 'subject_end', 'target_start', 'target_end']
        feature_dataset = feature_dataset.filter(lambda example: set(probe_indices_keys).issubset(set(example.keys())))

        # clean up dataset columns and datatypes
        if dataset_config.n_sequences > 0:
            feature_dataset = feature_dataset.select(range(dataset_config.n_sequences))

        feature_dataset = feature_dataset.remove_columns(['relation_prefix', 'prompt', 'subject'])
        feature_dataset.set_format(type="torch", columns=['tokens'] + probe_indices_keys,
                                   output_all_columns=True)

        if cache:
            self.save(dataset_config, feature_dataset)

        return feature_dataset
