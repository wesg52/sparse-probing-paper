from .common import *
from transformer_lens import utils
import numpy as np

COMPOUND_WORDS = [
    ('high', 'school'),
    ('living', 'room'),
    ('social', 'security'),
    ('credit', 'card'),
    ('blood', 'pressure'),
    ('prime', 'factors'),
    ('social', 'media'),
    ('gene', 'expression'),
    ('control', 'group'),
    ('magnetic', 'field'),
    ('cell', 'lines'),
    ('trial', 'court'),
    ('second', 'derivative'),
    ('north', 'america'),
    ('human', 'rights'),
    ('side', 'effects'),
    ('public', 'health'),
    ('federal', 'government'),
    ('third', 'party'),
    ('clinical', 'trials'),
    ('mental', 'health'),
]


class BigramFeatureDataset(FeatureDataset):
    def __init__(self, name, features):
        self.name = name
        self.features = features

    def make(self, dataset_config, args, raw_dataset, tokenizer, cache=True):
        tokenized_ds = utils.tokenize_and_concatenate(raw_dataset, tokenizer)
        all_tokens = tokenized_ds['tokens'][:, 1:].flatten().numpy()
        decoded_vocab = {
            tokenizer.decode(tix): tix
            for tix in tokenizer.get_vocab().values()
        }

        dataset_size = args.get('dataset_size', 8_000)
        target_positive_fraction = args.get('target_positive_fraction', 0.2)
        ctx_len = args.get('seq_len', 24)

        target_positive = int(dataset_size * target_positive_fraction)
        target_negative = dataset_size - target_positive

        bigram_datasets = []
        for first, second in self.features:
            compound_first_tokens = set(
                [v for k, v in decoded_vocab.items() if k.lower().strip() == first])
            compound_second_tokens = set(
                [v for k, v in decoded_vocab.items() if k.lower().strip() == second])

            first_indicator = np.isin(all_tokens, list(compound_first_tokens))
            second_indicator = np.isin(
                all_tokens, list(compound_second_tokens))

            first_occurences = np.where(first_indicator)[0]
            second_occurences = np.where(second_indicator)[0]

            bigram_indicator = np.isin(
                all_tokens[second_occurences - 1], list(compound_first_tokens))
            bigram_occurences = second_occurences[
                np.where(bigram_indicator)[0]]
            not_first_and_second_occurences = second_occurences[
                np.where(~bigram_indicator)[0]]
            first_and_not_second_occurences = first_occurences[np.where(
                ~second_indicator[first_occurences + 1])[0]] + 1

            n_pos = min(len(bigram_occurences), target_positive)

            pos_ixs = np.random.choice(bigram_occurences, n_pos, replace=False)
            not_first_neg_ixs = np.random.choice(
                not_first_and_second_occurences, int(target_negative / 2), replace=False)
            not_second_neg_ixs = np.random.choice(
                first_and_not_second_occurences, int(target_negative / 2), replace=False)

            all_ixs = np.sort(np.concatenate(
                [pos_ixs, not_first_neg_ixs, not_second_neg_ixs]))

            token_tensor = np.vstack([
                all_tokens[ix+1-ctx_len: ix+1] for ix in all_ixs
            ])

            feature_name = [f'{first}-{second}' for _ in range(len(all_ixs))]

            pos_set = set(pos_ixs)
            not_first_set = set(not_first_neg_ixs)

            label = [
                'bigram' if ix in pos_set
                else ('missing_first' if ix in not_first_set else 'missing_second')
                for ix in all_ixs
            ]

            ds = datasets.Dataset.from_dict({
                'tokens': token_tensor,
                'label': label,
                'feature_name': feature_name
            })
            ds.set_format(type="torch")

            bigram_datasets.append(ds)
            print(f'Finished processing {first}-{second}')

        full_ds = datasets.concatenate_datasets(bigram_datasets)

        if cache:
            self.save(dataset_config, full_ds)

        return full_ds

    def prepare_dataset(self, exp_cfg):
        full_ds = self.load(exp_cfg.dataset_cfg)
        ctx_len = exp_cfg.dataset_cfg.ctx_len
        offset = -2 if exp_cfg.probe_next_token_feature else -1
        feature_datasets = {}
        for first, second in self.features:
            feature_name = f'{first}-{second}'
            feature_ixs = np.array(full_ds['feature_name']) == feature_name
            feature_subset = full_ds.select(np.where(feature_ixs)[0])

            label_arr = np.array(feature_subset['label'])
            label = (label_arr == 'bigram').astype(int) * 2 - 1
            indices = np.arange(1, len(feature_subset) + 1) * ctx_len + offset
            # assumes features are contiguous
            subset_offset = np.min(np.where(feature_ixs)[0])
            indices += subset_offset * ctx_len

            feature_datasets[feature_name] = (indices, label)

        return full_ds, feature_datasets
