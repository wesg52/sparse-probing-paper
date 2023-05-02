import math
import copy
import numpy as np
from collections import defaultdict
import spacy_alignments as tokenizations
from .common import *

EWT_FEATURE_CATEGORIES = {
    'Number': {1: 'Sing', 2: 'Plur'},
    'Mood': {1: 'Ind', 2: 'Imp'},  # Just on Imp
    'Tense': {1: 'Pres', 2: 'Past'},
    'VerbForm': {1: 'Fin', 2: 'Inf', 3: 'Ger', 4: 'Part'},
    'PronType': {1: 'Art', 2: 'Dem', 3: 'Prs', 4: 'Rel', 5: 'Int'},
    'Person': {1: '1', 2: '2', 3: '3'},
    'NumType': {1: 'Card'},
    'Voice': {1: 'Pass'},
    'Gender': {1: 'Masc', 2: 'Fem', 3: 'Neut'},
    'eos': {1: True},
    'first_eos': {1: True},
}
POS_TAGS = {
    1: 'NOUN', 2: 'PUNC', 3: 'ADP', 4: 'NUM', 5: 'SYM', 6: 'SCONJ', 7: 'ADJ', 9: 'DET',
    10: 'CCONJ', 11: 'PROPN', 12: 'PRON', 13: 'X', 15: 'ADV', 16: 'INTJ', 17: 'VERB', 18: 'AUX'}
DEP_TAGS = {
    1: 'acl', 2: 'acl:relcl', 3: 'advcl', 4: 'advmod', 5: 'amod', 6: 'appos', 7: 'aux', 8: 'aux:pass', 9: 'case', 10: 'cc', 11: 'ccomp', 12: 'compound', 13: 'conj', 14: 'cop',
    15: 'det', 16: 'flat', 17: 'list', 18: 'mark', 19: 'nmod', 20: 'nmod:poss', 21: 'nsubj', 22: 'nsubj:pass', 23: 'nummod', 24: 'obj', 25: 'obl', 26: 'parataxis', 27: 'punct', 28: 'root', 29: 'xcomp'}
CATEGORICAL_FEATURE_MAPPING = {**EWT_FEATURE_CATEGORIES, **{
    'upos': POS_TAGS, 'dep': DEP_TAGS}}


def get_sentence_features(raw_dataset, model):
    feature_rows = {f: [] for f in list(EWT_FEATURE_CATEGORIES.keys())}
    upos_rows = []
    dep_rows = []
    head_rows = []
    valid_end_of_sentence = []
    within_compound_token_ix_rows = []
    max_compound_token_ix_rows = []
    model_tokenization = []
    for ix in range(len(raw_dataset)):
        # leading space for tokenization; replace for entry 912 in test set
        text = ' ' + raw_dataset[ix]['text'].replace('\xa0', ' ')
        idx = raw_dataset[ix]['idx']
        all_tokens = raw_dataset[ix]['tokens']
        lemmas = raw_dataset[ix]['lemmas']
        upos = raw_dataset[ix]['upos']
        feats = raw_dataset[ix]['feats']
        head = raw_dataset[ix]['head']
        deps = raw_dataset[ix]['deprel']
        misc = raw_dataset[ix]['misc']

        # tokenize sentence
        model_token_strs = model.to_str_tokens(text)[1:]
        model_tokens = model.to_tokens(text)[0, 1:].numpy()
        assert len(model_token_strs) == len(model_tokens)
        model_tokenization.append(model_tokens)

        # EWT splits contracted words and adds the two tokens to the token list;
        # sometimes copies implied words. We remove these to make the tokenization
        # (more) reversible.
        valid_ds_indices, valid_ds_tokens = zip(
            *[(jx, t) for jx, t in enumerate(all_tokens)
                if (lemmas[jx] != '_' or t == '_') and ('CopyOf' not in misc[jx])]
        )
        assert text.replace(' ', '') == ''.join(
            valid_ds_tokens).replace(' ', '')

        # add leading space for alignment
        valid_ds_tokens = list(valid_ds_tokens)
        valid_ds_tokens[0] = ' ' + valid_ds_tokens[0]

        # perform token alignment taking into account deleted tokens
        model2ds, ds2model = tokenizations.get_alignments(
            model_token_strs, valid_ds_tokens)
        reindexed_model2ds = [[valid_ds_indices[ds_t] for ds_t in model_t]
                              for model_t in model2ds]

        # collect features from EWT and map to model tokens
        # +1 to preserve 0 being feature absence
        upos_rows.append([upos[t[0]] + 1 if len(t) > 0 else 0
                          for t in reindexed_model2ds])
        dep_rows.append([deps[t[0]] if len(t) > 0 else 0
                         for t in reindexed_model2ds])
        head_rows.append([int(head[t[0]]) if len(t) > 0 else 0
                          for t in reindexed_model2ds])

        model_feats = [eval(feats[t[0]]) if len(t) > 0 else None
                       for t in reindexed_model2ds]
        for f in feature_rows:
            feature_rows[f].append([
                token_feat[f]
                if token_feat is not None and f in token_feat else None
                for token_feat in model_feats
            ])
        target_eos = len(model_token_strs) - 1
        valid_end_of_sentence.append([
            (ix == target_eos) and (('.' in s) or ('!' in s) or ('?' in s))
            for ix, s in enumerate(model_token_strs)
        ])

        # get information about how tokenization differs from EWT
        intra_token_ix = np.zeros(len(model2ds))
        intra_token_max_ix = np.zeros(len(model2ds))
        for ds_ix in ds2model:
            for m_ix in ds_ix:
                intra_token_ix[m_ix] += 1
                intra_token_max_ix[m_ix] = len(ds_ix)
        within_compound_token_ix_rows.append(
            intra_token_ix.astype(int).tolist())
        max_compound_token_ix_rows.append(
            intra_token_max_ix.astype(int).tolist())

    feature_rows['upos'] = upos_rows
    feature_rows['dep'] = dep_rows
    feature_rows['head'] = head_rows
    feature_rows['eos'] = valid_end_of_sentence
    feature_rows['within_compound_token_ix'] = within_compound_token_ix_rows
    feature_rows['max_compound_token_ix'] = max_compound_token_ix_rows
    feature_rows['tokens'] = model_tokenization

    return feature_rows


def convert_sentence_features_to_sequence_features(train_dataset, feature_rows, ctx_len):
    # organize sentences (and the corresponding features) into documents
    doc_map = defaultdict(list)
    for ix, id in enumerate(train_dataset['idx']):
        doc_map[id[:-5]].append(ix)

    # then split documents into appropriately sized sequences for the model
    seq_features = {f: [] for f in feature_rows}
    seq_features['doc_id'] = []
    for k, doc_seq_ixs in doc_map.items():
        sent_sizes = np.array([len(feature_rows['tokens'][ix])
                              for ix in doc_seq_ixs])
        seq_id = np.cumsum(sent_sizes) // (ctx_len - 1)
        n_seqs = seq_id.max() + 1
        seq_bounds = [0] + list(np.where(seq_id[1:] !=
                                seq_id[:-1])[0] + 1) + [len(doc_seq_ixs)]
        for i in range(n_seqs):
            seq_ixs = doc_seq_ixs[seq_bounds[i]:seq_bounds[i+1]]
            for f in feature_rows:
                if f == 'head':
                    continue
                seq_feature = [0]  # for BOS token
                for ix in seq_ixs:
                    seq_feature.extend(feature_rows[f][ix])
                seq_features[f].append(seq_feature)

            # handle head indexing (which is relative to the sequence)
            seq_heads = [0]  # for BOS token
            offset = 1
            for ix in seq_ixs:
                seq_heads.extend(
                    [h + offset for h in feature_rows['head'][ix]])
                offset += len(feature_rows['head'][ix])
            seq_features['head'].append(seq_heads)
            seq_features['doc_id'].append(k)

    for f in seq_features:
        if f == 'doc_id':
            continue
        # pad/crop sequences to the ctx_len
        seq_features[f] = [s[:ctx_len] for s in seq_features[f]]
        seq_features[f] = [
            s + [0 for _ in range((ctx_len - len(s)))] for s in seq_features[f]]

        # perform categorical mapping
        if f in CATEGORICAL_FEATURE_MAPPING:
            if f == 'upos':
                continue  # upos already integer categorical
            inv_mapping = {v: k for k,
                           v in CATEGORICAL_FEATURE_MAPPING[f].items()}

            def key_safe_mapping(x): return inv_mapping.get(x, 0)
            seq_features[f] = [list(map(key_safe_mapping, s))
                               for s in seq_features[f]]

    # special processing for eos
    seq_features['eos'] = [
        [1 if t == True else 0 for t in s]
        for s in seq_features['eos']
    ]
    min_eos = [s.index(1) if 1 in s else 0 for s in seq_features['eos']]
    seq_features['first_eos'] = [
        [1 if t == min_eos[i] else 0
            for t in range(len(seq_features['eos'][i]))]
        for i in range(len(min_eos))
    ]

    return seq_features


def make_preprocessed_ewt_dataset(model, ctx_len=512):
    dataset_splits = []
    for split in ['train', 'validation', 'test']:
        raw_dataset = datasets.load_dataset(
            "universal_dependencies", "en_ewt", split=split, streaming=False)
        raw_dataset = raw_dataset.filter(lambda x: len(x['tokens']) > 1)

        sentence_features = get_sentence_features(raw_dataset, model)
        sequence_features = convert_sentence_features_to_sequence_features(
            raw_dataset, sentence_features, ctx_len)

        ewt_split_ds = datasets.Dataset.from_dict(sequence_features)
        ewt_split_ds = ewt_split_ds.add_column(
            'split', [split for _ in range(len(ewt_split_ds))])
        ewt_split_ds = ewt_split_ds.add_column(
            'position', [np.arange(ctx_len) for _ in range(len(ewt_split_ds))])

        dataset_splits.append(ewt_split_ds)

    ewt_ds = datasets.concatenate_datasets(dataset_splits)
    ewt_ds.set_format('torch')

    save_name = f'preprocessed_ewt_{ctx_len}.hf'
    save_path = os.environ.get('HF_DATASETS_CACHE', 'downloads')
    ewt_ds.save_to_disk(os.path.join(save_path, save_name))


def make_probing_feature(valid_positives, valid_negatives, args, priority=None):
    n, ctx_len = valid_positives.shape

    valid_positives = valid_positives.flatten()
    valid_negatives = valid_negatives.flatten()

    n_pos = valid_positives.sum()
    n_neg = valid_negatives.sum()

    target_dataset_size = args.get('dataset_size', 25_000)
    target_pos_frac = args.get('target_positive_fraction', 0.2)

    max_n_pos = int(target_dataset_size * target_pos_frac)
    n_pos_indices = min(n_pos, max_n_pos)
    n_neg_indices = int(min(target_dataset_size - n_pos_indices, n_neg))

    if priority is None:
        positive_probe_indices = np.random.choice(
            np.where(valid_positives)[0], n_pos_indices, replace=False)

        negative_probe_indices = np.random.choice(
            np.where(valid_negatives)[0], n_neg_indices, replace=False)
    else:
        # Take the top positive and negative indices by index_priority
        priority_positive_mask = np.isin(
            priority, np.where(valid_positives)[0])
        positive_probe_indices = priority[priority_positive_mask][:n_pos_indices]

        priority_negative_mask = np.isin(
            priority, np.where(valid_negatives)[0])
        negative_probe_indices = priority[priority_negative_mask][:n_neg_indices]

    probe_indices = np.zeros(n*ctx_len, dtype=int)
    probe_indices[positive_probe_indices] = 1
    probe_indices[negative_probe_indices] = -1
    probe_indices = probe_indices.reshape(n, ctx_len)

    hf_probe_indices_column = [
        np.nonzero(probe_indices[i])[0] for i in range(n)]
    hf_probe_classes_column = [
        probe_indices[i][hf_probe_indices_column[i]] for i in range(n)]

    return hf_probe_indices_column, hf_probe_classes_column


def get_valid_index_labels(ewt_dataset, feature_category, feature_class_id, args):
    k = args.get('ignore_first_k', 5)

    feature_labels = ewt_dataset[feature_category].numpy()
    non_pad_tokens = (ewt_dataset['tokens'] > 1).numpy()
    valid_tokens = non_pad_tokens & (ewt_dataset['position'] >= k).numpy()

    if feature_category in {'Number', 'Mood', 'Tense'}:
        # probe on plural, Imp, Past vs singular, Ind, Pres
        valid_pos = feature_labels == 2
        valid_neg = feature_labels == 1
    elif feature_category == 'NumType':
        valid_pos = feature_labels == 1
        valid_neg = valid_tokens & ~valid_pos
    elif feature_category == 'Voice':
        valid_pos = feature_labels == 1
        valid_neg = (ewt_dataset['upos'] == 17).numpy() & ~valid_pos
    elif feature_category == 'eos':
        valid_pos = feature_labels == 1
        valid_neg = valid_tokens & ~valid_pos
    elif feature_category == 'first_eos':
        valid_pos = feature_labels == 1
        valid_neg = (ewt_dataset['eos'] == 1).numpy() & ~valid_pos
    else:
        valid_pos = feature_labels == feature_class_id
        valid_neg = (feature_labels != 0) & ~valid_pos

    return valid_pos & valid_tokens, valid_neg & valid_tokens


class LinguisticFeatureDataset(FeatureDataset):
    def __init__(self, name):
        self.name = name

    def make(self, dataset_config, args, ewt_dataset):
        feature_dataset = copy.deepcopy(ewt_dataset)

        # random index priority to faciliate more overlap of postive and negative samples
        n_tokens = len(feature_dataset['tokens'].flatten())
        index_priority = np.random.permutation(np.arange(n_tokens))

        # first binarize the categorical features
        unrestricted_categories = [  # categories where can simply do one vs all
            'upos', 'dep', 'VerbForm', 'PronType', 'Person', 'Gender']
        for feature_category in unrestricted_categories:
            feature_classes = CATEGORICAL_FEATURE_MAPPING[feature_category]
            for feature_class_id, feature_name in feature_classes.items():
                print(feature_category, feature_name)
                valid_positives, valid_negatives = get_valid_index_labels(
                    ewt_dataset, feature_category, feature_class_id, args)

                hf_probe_indices_column, hf_probe_classes_column = make_probing_feature(
                    valid_positives, valid_negatives, args, index_priority)

                column_name_prefix = f'{feature_category}_{feature_name}'
                feature_dataset = feature_dataset.add_column(
                    f'{column_name_prefix}|probe_indices', hf_probe_indices_column)
                feature_dataset = feature_dataset.add_column(
                    f'{column_name_prefix}|probe_classes', hf_probe_classes_column)

        # then binarize categorical features with restrictions on negatives
        restricted_categories = ['Number', 'Mood',
                                 'Tense', 'NumType', 'Voice', 'eos', 'first_eos']
        for feature_category in restricted_categories:
            feature_classes = CATEGORICAL_FEATURE_MAPPING[feature_category]
            valid_positives, valid_negatives = get_valid_index_labels(
                ewt_dataset, feature_category, 0, args)  # we don't use the feature class id here

            hf_probe_indices_column, hf_probe_classes_column = make_probing_feature(
                valid_positives, valid_negatives, args, index_priority)

            feature_name = feature_classes[max(feature_classes.keys())]
            column_name_prefix = f'{feature_category}_{feature_name}'
            feature_dataset = feature_dataset.add_column(
                f'{column_name_prefix}|probe_indices', hf_probe_indices_column)
            feature_dataset = feature_dataset.add_column(
                f'{column_name_prefix}|probe_classes', hf_probe_classes_column)

        self.save(dataset_config, feature_dataset)

    def prepare_dataset(self, exp_cfg):
        """Process saved HF dataset to be consumed by probes.

        Returns: tokenized dataset, feature dataset with structure:
            {feature: (indices, classes)}"""
        dataset = self.load(exp_cfg.dataset_cfg)
        feature_names = [
            name.split('|')[0] for name in dataset.column_names
            if name.endswith('|probe_indices')
        ]
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
