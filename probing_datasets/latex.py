import re
from numba import jit
from tqdm import tqdm
from .common import *

# Thanks GPT4
TITLE_REGEX = re.compile(
    r'^title:(?:\s+\||\s+\'| )(.+?)(?:\'|\\|\n|$)(?:\n|$)',
    re.MULTILINE | re.DOTALL
)
AUTHOR_REGEX = re.compile(
    r'author:((?:(?:(?!author:|title:|bibliography:|date:|address:|---).)*\n)*)',
    re.MULTILINE
)
ABSTRACT_REGEX = re.compile(
    r'abstract:((?:(?:(?!author:|title:|bibliography:|date:|address:|---).)*\n)*)',
    re.MULTILINE
)
SUPERSCRIPT_REGEX = re.compile(r'\^{(.*?)}', re.MULTILINE)
SUBSCRIPT_REGEX = re.compile(r'_{(.*?)}', re.MULTILINE)
CITATION_REGEX = re.compile(r'\[@(.*?)\]', re.DOTALL | re.MULTILINE)
FRACTION_REGEX = re.compile(r'\\frac{([^}]+)}{([^}]+)}')


def get_superscript_tokens(tokens, c2t, latex_str):
    is_superscript = np.zeros_like(tokens)
    for sup in SUPERSCRIPT_REGEX.finditer(latex_str):
        is_superscript[c2t[sup.start(1)]:c2t[sup.end(1)]] = 1
    return is_superscript


def get_subscript_tokens(tokens, c2t, latex_str):
    is_subscript = np.zeros_like(tokens)
    for sub in SUBSCRIPT_REGEX.finditer(latex_str):
        is_subscript[c2t[sub.start(1)]:c2t[sub.end(1)]] = 1
    return is_subscript


def get_reference_tokens(tokens, c2t, latex_str):
    is_reference = np.zeros_like(tokens)
    for ref in CITATION_REGEX.finditer(latex_str):
        is_reference[c2t[ref.start(1)]:c2t[ref.end(1)]] = 1
    return is_reference


def get_fraction_tokens(tokens, c2t, latex_str):
    matches = list(FRACTION_REGEX.finditer(latex_str))
    is_frac = np.zeros_like(tokens)
    is_numerator = np.zeros_like(tokens)
    is_denominator = np.zeros_like(tokens)
    for frac in matches:
        is_frac[c2t[frac.start(1)-1]:c2t[frac.end(2)+1]] = 1
        is_numerator[c2t[frac.start(1)]:c2t[frac.end(1)]] = 1
        is_denominator[c2t[frac.start(2)]:c2t[frac.end(2)]] = 1
    return is_frac, is_numerator, is_denominator


def get_title_tokens(tokens, c2t, latex_str):
    is_title = np.zeros_like(tokens)
    title = TITLE_REGEX.search(latex_str)
    if title is not None:
        end = title.end(1) if title.end(1) - title.start(1) < 50 \
            else title.start(1) + 50
        end = min(end, len(latex_str) - 1)
        is_title[c2t[title.start(1)]: c2t[end]] = 1
    return is_title


def get_abstract_tokens(tokens, c2t, latex_str):
    is_abstract = np.zeros_like(tokens)
    abs = ABSTRACT_REGEX.search(latex_str)
    if abs is not None:
        end = abs.end(1) if abs.end(1) - abs.start(1) < 400 \
            else abs.start(1) + 250
        end = min(end, len(latex_str) - 1)
        is_abstract[c2t[abs.start(1)]: c2t[end]] = 1
    return is_abstract


def get_author_tokens(tokens, c2t, latex_str):
    is_author = np.zeros_like(tokens)
    author = AUTHOR_REGEX.search(latex_str)
    if author is not None:
        end = author.end(1) if author.end(1) - author.start(1) < 200 \
            else author.start(1) + 75
        end = min(end, len(latex_str) - 1)
        is_author[c2t[author.start(1)]: c2t[end]] = 1
    return is_author


def extract_math_spans(latex_str):
    inline_math_spans = []
    display_math_spans = []

    inline_math_start = None
    display_math_start = None

    in_escape = False
    in_inline = False
    in_display = False
    skip_next = False

    for i, char in enumerate(latex_str):
        if skip_next:
            skip_next = False
            continue

        if in_escape:
            in_escape = False
            continue

        if char == "\\":
            in_escape = True
            continue

        if in_inline:
            if char == "$":
                in_inline = False
                inline_math_spans.append((inline_math_start, i-1))
                inline_math_start = None

        elif in_display:
            if char == "$" and latex_str[i + 1] == "$":
                in_display = False
                display_math_spans.append((display_math_start, i-1))
                display_math_start = None
                skip_next = True

        else:
            if char == "$":
                if latex_str[i + 1] == "$":
                    display_math_start = i + 2
                    in_display = True
                else:
                    inline_math_start = i + 1
                    in_inline = True

    return inline_math_spans, display_math_spans


def get_math_tokens(tokens, c2t, latex_str):
    is_math = np.zeros_like(tokens)
    is_inline_math = np.zeros_like(tokens)
    is_display_math = np.zeros_like(tokens)
    start_math = np.zeros_like(tokens)
    end_math = np.zeros_like(tokens)

    inline_math_spans, display_math_spans = extract_math_spans(latex_str)

    for span in inline_math_spans:
        is_math[c2t[span[0]]:c2t[span[1]]] = 1
        is_inline_math[c2t[span[0]]:c2t[span[1]]] = 1
        start_math[c2t[span[0] - 1]] = 1
        end_math[c2t[span[1] + 1]] = 1

    for span in display_math_spans:
        is_math[c2t[span[0]]:c2t[span[1]]] = 1
        is_display_math[c2t[span[0]]:c2t[span[1]]] = 1
        start_math[c2t[span[0] - 2]] = 1
        end_math[c2t[span[1] + 2]] = 1

    return is_math, is_inline_math, is_display_math, start_math, end_math


@jit(nopython=True)
def create_char_to_token_map(offsets, n):
    char_to_token = np.ones(n, dtype=np.int32) * -1
    # Create a mapping from character index to token index
    for token_idx, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        char_to_token[start: end] = token_idx
    return char_to_token


FEATURES = [
    'is_frac', 'is_numerator', 'is_denominator', 'is_title', 'is_abstract', 'is_author',
    'is_subscript', 'is_superscript', 'is_reference', 'is_math', 'is_inline_math', 'is_display_math',
    'start_math', 'end_math'
]
NON_MIRRORED_FEATURES = [
    'is_title', 'is_author', 'is_abstract', 'is_reference', 'is_math', 'is_frac'
]
MIRRORED_FEATURES = {
    'is_denominator': 'is_numerator',
    'is_subscript': 'is_superscript',
    'is_inline_math': 'is_display_math',
    'start_math': 'end_math'
}
MIRRORED_FEATURES = {
    **MIRRORED_FEATURES,
    **{k: v for v, k in MIRRORED_FEATURES.items()}
}


def make_probing_feature(args, ix_priority, valid_ix_dict, key):
    valid_positives = valid_ix_dict[key] > 0
    n, ctx_len = valid_positives.shape

    valid_negatives = ~valid_positives
    valid_negatives = valid_negatives & (
        np.arange(ctx_len) > args.get('ignore_first_k', 5))[None, :]

    valid_positives = valid_positives.flatten()
    valid_negatives = valid_negatives.flatten()

    n_pos = valid_positives.sum()
    n_neg = valid_negatives.sum()

    target_dataset_size = args.get('dataset_size', 20_000)
    target_pos_frac = args.get('target_positive_fraction', 0.25)

    max_n_pos = int(target_dataset_size * target_pos_frac)
    n_pos_indices = min(n_pos, max_n_pos)
    n_neg_indices = int(min(target_dataset_size - n_pos_indices, n_neg))

    priority_positive_mask = np.isin(
        ix_priority, np.where(valid_positives)[0])
    positive_probe_indices = ix_priority[priority_positive_mask][:n_pos_indices]

    if key in set(NON_MIRRORED_FEATURES):
        priority_negative_mask = np.isin(
            ix_priority, np.where(valid_negatives)[0])
        negative_probe_indices = ix_priority[priority_negative_mask][:n_neg_indices]

    elif key in MIRRORED_FEATURES:
        # if feature with natural mirror negative, have half negatives be the mirror
        # and the other half be random tokens
        valid_mirrored_negatives = (
            valid_ix_dict[MIRRORED_FEATURES[key]] > 0).flatten()
        valid_mirrored_negatives &= valid_negatives

        mir_n_neg = n_neg_indices // 2
        priority_mirror_negative_mask = np.isin(
            ix_priority, np.where(valid_mirrored_negatives)[0])
        mirror_neg_probe_indices = ix_priority[priority_mirror_negative_mask][:mir_n_neg]

        priority_negative_mask = np.isin(
            ix_priority, np.where(valid_negatives)[0])
        non_mirror_neg_probe_indices = ix_priority[priority_negative_mask][:mir_n_neg]

        negative_probe_indices = np.concatenate(
            [mirror_neg_probe_indices, non_mirror_neg_probe_indices])

    probe_indices = np.zeros(n*ctx_len, dtype=int)
    probe_indices[positive_probe_indices] = 1
    probe_indices[negative_probe_indices] = -1
    probe_indices = probe_indices.reshape(n, ctx_len)

    hf_probe_indices_column = [
        np.nonzero(probe_indices[i])[0] for i in range(n)]
    hf_probe_classes_column = [
        probe_indices[i][hf_probe_indices_column[i]] for i in range(n)]

    return hf_probe_indices_column, hf_probe_classes_column


class LatexFeatureDataset(FeatureDataset):
    def __init__(self):
        self.name = "latex"

    def make(self, dataset_config, args, raw_dataset, tokenizer, cache=True):
        n = dataset_config.n_sequences \
            if dataset_config.n_sequences > 0 else len(raw_dataset['text'])
        d = dataset_config.ctx_len
        bos_offset = 1 if args.get('add_bos', True) else 0

        token_matrix = np.zeros((n, d), dtype=np.int32)
        offset_matrix = np.zeros((n, d+1), dtype=np.int32)
        texts = []
        i = 0
        # first light quality filter then tokenize
        for raw_text in tqdm(raw_dataset['text'], mininterval=3):
            reversible_text = tokenizer.decode(tokenizer.encode(raw_text))
            if reversible_text.count('$') % 2 != 0:
                continue
            encoded = tokenizer.encode_plus(
                reversible_text, return_offsets_mapping=True)
            if len(encoded['input_ids']) < d:
                continue
            token_matrix[i, bos_offset:] = encoded["input_ids"][:d-bos_offset]
            char_offset = np.array(encoded["offset_mapping"])
            char_offset = char_offset[:d-bos_offset, 1]
            offset_matrix[i, bos_offset+1:] = char_offset
            texts.append(reversible_text)
            i += 1
            if i >= n:
                break
        n = i
        print(f'Finished filtering and tokenizing {n} sequences.')

        # extract token features for each of the sequences
        valid_feature_matrices = {
            k: np.zeros((n, d), dtype=np.int32) for k in FEATURES
        }
        for i in tqdm(range(n), mininterval=1):
            tokens = token_matrix[i]
            offsets = offset_matrix[i]
            reversible_text = texts[i]

            char_to_token = create_char_to_token_map(
                offsets, len(reversible_text))

            title_tokens = get_title_tokens(
                tokens, char_to_token, reversible_text)
            abstract_tokens = get_abstract_tokens(
                tokens, char_to_token, reversible_text)
            author_tokens = get_author_tokens(
                tokens, char_to_token, reversible_text)

            reference_tokens = get_reference_tokens(
                tokens, char_to_token, reversible_text)
            subscript_tokens = get_subscript_tokens(
                tokens, char_to_token, reversible_text)
            superscript_tokens = get_superscript_tokens(
                tokens, char_to_token, reversible_text)
            math_info = get_math_tokens(tokens, char_to_token, reversible_text)
            is_math, is_inline_math, is_display_math, start_math, end_math = math_info
            frac_info = get_fraction_tokens(
                tokens, char_to_token, reversible_text)
            is_frac, is_numerator, is_denominator = frac_info

            valid_feature_matrices['is_title'][i] = title_tokens[:d]
            valid_feature_matrices['is_abstract'][i] = abstract_tokens[:d]
            valid_feature_matrices['is_author'][i] = author_tokens[:d]

            valid_feature_matrices['is_subscript'][i] = subscript_tokens[:d]
            valid_feature_matrices['is_superscript'][i] = superscript_tokens[:d]
            valid_feature_matrices['is_reference'][i] = reference_tokens[:d]
            valid_feature_matrices['is_math'][i] = is_math[:d]
            valid_feature_matrices['is_inline_math'][i] = is_inline_math[:d]
            valid_feature_matrices['is_display_math'][i] = is_display_math[:d]
            valid_feature_matrices['start_math'][i] = start_math[:d]
            valid_feature_matrices['end_math'][i] = end_math[:d]
            valid_feature_matrices['is_frac'][i] = is_frac[:d]
            valid_feature_matrices['is_numerator'][i] = is_numerator[:d]
            valid_feature_matrices['is_denominator'][i] = is_denominator[:d]

        feature_dataset = datasets.Dataset.from_dict(
            {'tokens': token_matrix[:n]})

        print('Finished extracting features')
        # create the dataset by choosing probe indices
        index_priority = np.random.permutation(np.arange(n*d))
        for k in valid_feature_matrices.keys():
            valid_indices = valid_feature_matrices[k] > 0
            indices, labels = make_probing_feature(
                args, index_priority, valid_feature_matrices, k)

            feature_dataset = feature_dataset.add_column(
                f'{k}|probe_indices', indices)
            feature_dataset = feature_dataset.add_column(
                f'{k}|probe_classes', labels)
            feature_dataset = feature_dataset.add_column(
                f'{k}|valid_indices', [row for row in valid_indices])

            print(f'Finished constructing {k} probe dataset')

        feature_dataset.set_format('torch')

        if cache:
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
