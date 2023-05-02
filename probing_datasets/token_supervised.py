import re
import os
import torch
import numpy as np
from functools import partial
from scipy import sparse
from .common import *

# TODO add numeric words


class PlainTextFeatureDataset(FeatureDataset):
    def __init__(self, name, feature_collection):
        self.name = name
        self.feature_collection = feature_collection

    def make(self, dataset_config, args, raw_dataset, tokenizer, cache=True):
        tokenized_dataset = tokenize_and_concatenate_separate_subsequences(
            raw_dataset,
            tokenizer,
            max_length=dataset_config.ctx_len,
            add_bos_token=args.get('add_bos_token', True),
        )

        # Get decoded vocab to avoid unicode issues.
        vocab = tokenizer.get_vocab()
        vocab = {tokenizer.decode(tix): tix for tix in vocab.values()}

        # Determine which tokens have specific features.
        token_tensor = tokenized_dataset['tokens']
        feature_masks = []
        for feature_name, feature_condition in self.feature_collection:
            positive_tokens = torch.tensor([
                tix for string, tix in vocab.items() if feature_condition(string)
            ])
            feature_dataset = torch.isin(token_tensor, positive_tokens)
            feature_dataset = feature_dataset.numpy().astype(bool)

            feature_masks.append((feature_name, feature_dataset))

        ignore_first_k_tokens = args.get('ignore_first_k', 10)
        data_size = args.get('dataset_size', 10_000)
        target_positive_fraction = args.get('target_positive_fraction', 0.2)
        min_positive_fraction = args.get('min_positive_fraction', 0.05)
        n, ctx_len = token_tensor.shape

        # Determine which tokens are valid for probing.
        valid_tokens = torch.ones_like(token_tensor, dtype=bool)
        valid_tokens[:, :ignore_first_k_tokens] = False
        special_tokens = torch.tensor([
            tokenizer.pad_token_id,
            tokenizer.unk_token_id,
            tokenizer.eos_token_id,
            tokenizer.bos_token_id
        ])
        valid_tokens[torch.isin(token_tensor, special_tokens)] = False

        valid_indices = torch.arange(
            n*ctx_len).reshape(n, ctx_len)[valid_tokens].numpy()
        # Create an arbirtary prioritization of indices to increase token overlap
        # between classes to decrease memory requirements of activation dataset.
        index_priority = np.random.permutation(valid_indices)

        # Add probe indices and classes to HF dataset
        for feature_name, feature_dataset in feature_masks:
            valid_feature_dataset = feature_dataset.flatten()[valid_indices]
            positive_frac = valid_feature_dataset.mean()
            if positive_frac * len(valid_feature_dataset) < min_positive_fraction * data_size:
                continue

            required_positives = int(
                min(max(positive_frac, target_positive_fraction), 0.5) * data_size)
            required_negatives = data_size - required_positives

            valid_positive_indices = valid_indices[valid_feature_dataset]
            valid_negative_indices = valid_indices[~valid_feature_dataset]

            # Take the top positive and negative indices by index_priority
            priority_positive_mask = np.isin(
                index_priority, valid_positive_indices)
            positive_probe_indices = index_priority[priority_positive_mask][:required_positives]
            priority_negative_mask = np.isin(
                index_priority, valid_negative_indices)
            negative_probe_indices = index_priority[priority_negative_mask][:required_negatives]

            # Add indices to HF dataset
            probe_indices = np.zeros(n*ctx_len, dtype=int)
            probe_indices[positive_probe_indices] = 1
            probe_indices[negative_probe_indices] = -1
            probe_indices = probe_indices.reshape(n, ctx_len)

            hf_probe_indices_column = [
                np.nonzero(probe_indices[i])[0] for i in range(n)]
            hf_probe_classes_column = [
                probe_indices[i][hf_probe_indices_column[i]] for i in range(n)]

            tokenized_dataset = tokenized_dataset.add_column(
                f'{feature_name}|probe_indices', hf_probe_indices_column)
            tokenized_dataset = tokenized_dataset.add_column(
                f'{feature_name}|probe_classes', hf_probe_classes_column)

        tokenized_dataset.set_format('torch')

        if cache:
            self.save(dataset_config, tokenized_dataset)

        return tokenized_dataset

    def prepare_dataset(self, exp_cfg):
        """Process saved HF dataset to be consumed by probes.

        Returns: tokenized dataset, feature dataset with structure:
            {feature: (indices, classes)}"""
        dataset = self.load(exp_cfg.dataset_cfg)
        feature_names = [
            name.split('|')[0] for name in dataset.column_names
            if name.endswith('|probe_indices')
        ]  # Not equivalent to feature collection because some features
        # may not have enough data to be included in the dataset.
        feature_datasets = {name: {'indices': [], 'classes': []}
                            for name in feature_names}
        offset = -1 if exp_cfg.probe_next_token_feature else 0
        ctx_len = exp_cfg.dataset_cfg.ctx_len
        feature_datasets = {}
        for feature in feature_names:
            indices = np.concatenate([
                row.numpy() + (ix * ctx_len) + offset for ix, row
                in enumerate(dataset[f'{feature}|probe_indices'])
            ])
            classes = np.concatenate([
                row.numpy() for row in dataset[f'{feature}|probe_classes']
            ])
            feature_datasets[feature] = (indices, classes)

        return dataset, feature_datasets


DIGIT_RE = re.compile('\d')


def contains_digit(string):
    return DIGIT_RE.search(string) is not None


ALL_DIGITS_RE = re.compile('\s?\d+')


def all_digits(string):
    return ALL_DIGITS_RE.fullmatch(string) is not None


CAPITAL_RE = re.compile('[A-Z]+')


def contains_capital(string):
    return CAPITAL_RE.search(string) is not None


def leading_capital(string):
    return (len(string) > 1 and string[0] == " " and string[1].isupper()) \
        or (string[0] != " " and string[0].isupper())


ALL_CAPITAL_RE = re.compile('[\s]*[A-Z]+')


def all_capitals(string):
    invalid_all_caps = {'I', ' I', 'A', ' A'}
    return ALL_CAPITAL_RE.fullmatch(string) is not None and \
        string not in invalid_all_caps


WHITESPACE_RE = re.compile('[\s]')


def contains_whitespace(string):
    return WHITESPACE_RE.search(string) is not None


def has_leading_space(string):
    return string[0] == " "


def no_leading_space_and_loweralpha(string):
    # isalpha is false if contains space
    return string.isalpha() and string.islower()


ALL_WHITESPACE_RE = re.compile('[\s]+')


def contains_all_whitespace(string):
    return ALL_WHITESPACE_RE.fullmatch(string) is not None


def starts_with_whitespace_vowel(string):
    VOWELS = {'a', 'e', 'i', 'o', 'u'}
    return len(string) > 1 and (string[0] == ' ') and (string[1].lower() in VOWELS)


NOT_ALPHANUMERIC_RE = re.compile('[\W]+')


def is_not_alphanumeric(string):
    return NOT_ALPHANUMERIC_RE.fullmatch(string) is not None


# TODO(kharvey) look at tokenizer for others
MONITARY_RE = re.compile('[$€£¥฿₹]+')


def contains_monetary_character(string):
    return MONITARY_RE.search(string) is not None


HONORIFICS_RE = re.compile(r"\b(Mr|Ms|Mrs|Dr|Prof|Hon|Rev)\b")


def is_not_ascii(string):
    return not string.isascii()


def contains_honorific(string):
    return HONORIFICS_RE.search(string) is not None


LEFT_PUNCT_RE = re.compile('[\\[({<]')
RIGHT_PUNCT_RE = re.compile('[\\])}>]')


def is_left_punctuation(string):
    return LEFT_PUNCT_RE.search(string) is not None


def is_right_punctuation(string):
    return RIGHT_PUNCT_RE.search(string) is not None


### Major suffixes and prefixes ###
# see http://www.uefap.com/vocab/build/building.htm
# TODO: allow multiple for special case spellings (eg. ise and ive)
prefixes = {
    # Verbs
    're',
    'dis',
    'over',
    'un',
    'mis',
    'out',
    'be',
    'co',
    'de',
    'fore',
    'inter',
    'pre',
    'sub',
    'trans',
    'under'
    # Nouns
    'anti',
    'auto',
    'bi',
    'counter',
    'ex',
    'hyper',
    'in',
    'inter',
    'kilo',
    'mal',
    'mega',
    'mis',
    'mini',
    'mono',
    'neo',
    'out',
    'poly',
    'pseudo',
    'semi',
    'sub',
    'super',
    'tele',
    'tri',
    'ultra',
    'under',
    'vice',
    # Adjectives
    ('il', 'ir', 'im', 'in'),
    'non'
}


def starts_with_prefix(prefix, string):
    def check_prefix(p):
        return string[1:len(p)+1].lower() == f' {p}' or \
            string[:len(p)].lower() == p
    if isinstance(prefix, str):
        return check_prefix(prefix)
    else:  # if iterable, check for any match
        return any([check_prefix(p) for p in prefix])


prefix_conditions = [
    (f'starts_with_{prefix}', partial(starts_with_prefix, prefix))
    for prefix in prefixes
]


suffixes = {
    # Verbs
    ('ise', 'ize'),
    'ate',
    'fy',
    'en',
    # Nouns
    ('tion', 'sion'),
    'er',
    'ment',
    ('ant', 'ent'),
    'age',
    'al',
    ('ence', 'ance'),
    'ry',
    'ism',
    'ship',
    'age',
    'ity',
    'ness',
    'cy',
    # Adjectives
    'al',
    'ent',
    'ive',
    'ous',
    'ful',
    'less',
    'able'
}


def ends_with_suffix(suffix, string):
    if isinstance(suffix, str):
        return string[-len(suffix):].lower() == suffix
    else:  # if iterable, check for any match
        return any([string[-len(s):].lower() == s for s in suffix])


def create_suffix_prefix_string(s_or_p):
    if isinstance(s_or_p, str):
        return s_or_p
    else:
        return ','.join(s_or_p)


suffix_conditions = [
    (f'ends_with_{suffix}', partial(ends_with_suffix, suffix))
    for suffix in suffixes
]


true_binary_token_supervised_feature_conditions = [
    ('contains_digit', contains_digit),
    ('all_digits', all_digits),
    ('contains_capital', contains_capital),
    ('leading_capital', leading_capital),
    ('all_capitals', all_capitals),
    ('contains_whitespace', contains_whitespace),
    ('has_leading_space', has_leading_space),
    ('no_leading_space_and_loweralpha', no_leading_space_and_loweralpha),
    ('contains_all_whitespace', contains_all_whitespace),
    ('is_not_alphanumeric', is_not_alphanumeric),
    ('is_not_ascii', is_not_ascii),
]
