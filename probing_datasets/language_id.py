import os
import datasets
import torch
import numpy as np
from .common import *


CODE_LANGS = {
    'C++': 1,
    'C': 2,
    'Go': 3,
    'HTML': 4,
    'Java': 5,
    'JavaScript': 6,
    'PHP': 7,
    'Python': 8,
    'XML': 9
}

NATURAL_LANGS = {
    'bg': 1,
    'de': 2,
    'es': 3,
    'fr': 4,
    'lt': 5,
    'pl': 6,
    'sk': 7,
    'da': 8,
    'en': 9,
    'fi': 10,
    'it': 11,
    'nl': 12,
    'ro': 13,
    'sv': 14,
    'cs': 15,
    'el': 16,
    'et': 17,
    'hu': 18,
    'lv': 19,
    'pt': 20,
    'sl': 21
}

TOP_NATURAL_LANGS = {
    'es': 1,
    'en': 2,
    'fr': 3,
    'nl': 4,
    'it': 5,
    'el': 6,
    'de': 7,
    'pt': 8,
    'sv': 9
}


class LanguageIDFeatureDataset(FeatureDataset):

    def __init__(self, name, langs):
        self.name = name
        self.langs = langs

    def prepare_dataset(self, exp_cfg):
        """Convert categorial labels into binary labels for each language.

        Returns: tokenized_dataset, feature_datasets with structure:
            {lang: (indices, classes)}.

        ...except when exp_cfg.aggregation is not None. Then
        indices are valid_index mask with class per row."""
        dataset = self.load(exp_cfg.dataset_cfg)
        _, n = dataset['tokens'].shape

        feature_datasets = {k: {'indices': [], 'classes': []}
                            for k in self.langs.keys()}

        dataset_langs = np.array(dataset['lang'])

        if exp_cfg.activation_aggregation is None:
            valid_index_mask = torch.zeros_like(dataset['tokens'])
            for ix, valid_seq_indices in enumerate(dataset['probe_indices']):
                valid_index_mask[ix, valid_seq_indices] = 1
            valid_indices = np.where(valid_index_mask.flatten())[0]

            extended_dataset_langs = dataset_langs[:, None].repeat(
                n, axis=1).flatten()
            for ix, l in enumerate(self.langs):
                lang_label = (extended_dataset_langs == l).astype(int) * 2 - 1
                # same index_mask for all languages
                feature_datasets[l] = (valid_indices, lang_label)
            return dataset, feature_datasets

        else:  # each feature dataset is (size(n x ctx_len) index_mask, size(n) class)
            valid_index_mask = torch.zeros_like(dataset['tokens'])
            for ix, valid_seq_indices in enumerate(dataset['valid_indices']):
                valid_index_mask[ix, valid_seq_indices] = 1

            for ix, l in enumerate(self.langs):
                lang_label = (dataset_langs == l).astype(int) * 2 - 1
                # same index_mask for all languages
                feature_datasets[l] = (valid_index_mask, lang_label)

            return dataset, feature_datasets

    def make(self, dataset_config, args, raw_dataset, tokenizer, cache=True):

        n_probe_tokens = args.get('lang_id_n_tokens', 5)
        ignore_k = args.get('ignore_first_k', 100)

        tokenized_dataset = tokenize_and_concatenate_separate_subsequences(
            raw_dataset,
            tokenizer,
            max_length=dataset_config.ctx_len,
            add_bos_token=args.get('add_bos_token', True),
        )
        class_ids = [self.langs.get(lang, 0) for lang in raw_dataset['lang']]
        feature_dataset = tokenized_dataset.add_column('class_ids', class_ids)
        # filter out examples with unknown language and too short sequences.
        feature_dataset = feature_dataset.filter(lambda x: x['class_ids'] != 0)
        feature_dataset = feature_dataset.filter(
            lambda x: len(x['all_tokens']) > ignore_k + n_probe_tokens)

        if self.name == 'natural_lang_id':
            # some europarl documents are massive, so drop full tokenization and text.
            feature_dataset = feature_dataset.remove_columns('all_tokens')
            feature_dataset = feature_dataset.remove_columns('text')

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

            if eos - ignore_k > n_probe_tokens:
                valid_indices = np.arange(ignore_k, eos)
                probe_indices = sorted(np.random.choice(
                    valid_indices, size=n_probe_tokens, replace=False).tolist())
            else:
                probe_indices = []
                valid_indices = []
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


# EuroParl helpers
def cache_europarl_dataset():
    europarl_dataset = datasets.load_dataset(
        "the_pile", "europarl", split="train")
    langs = [eval(sample['meta'])['language']
             for sample in europarl_dataset]

    europarl_dataset = europarl_dataset.remove_columns(['meta'])
    europarl_dataset = europarl_dataset.add_column('lang', langs)
    save_path = os.path.join(
        os.environ['HF_DATASETS_CACHE'], 'the_pile', 'europarl.hf')
    europarl_dataset.save_to_disk(save_path)


NATURAL_LANGS_UNABBREVIATED = {
    'bg': 'Bulgarian',
    'de': 'German',
    'es': 'Spanish',
    'fr': 'French',
    'lt': 'Lithuanian',
    'pl': 'Polish',
    'sk': 'Slovak',
    'da': 'Danish',
    'en': 'English',
    'fi': 'Finnish',
    'it': 'Italian',
    'nl': 'Dutch',
    'ro': 'Romanian',
    'sv': 'Swedish',
    'cs': 'Czech',
    'el': 'Greek',
    'et': 'Estonian',
    'hu': 'Hungarian',
    'lv': 'Lativian',
    'pt': 'Portuguese',
    'sl': 'Slovenian'
}
