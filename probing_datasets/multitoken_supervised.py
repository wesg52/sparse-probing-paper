import re
import numpy as np
from .common import *


import regex

# https://stackoverflow.com/questions/26385984/recursive-pattern-in-regex
IN_QUOTES_REGEX = re.compile(r'[“"”](.*?)[”"]')

IN_PARENTHESES_REGEX = regex.compile(r"\(((?>[^()]+|(?R))*)\)")

IN_BRACKETS_REGEX = regex.compile(r"\[((?>[^\[\]]+|(?R))*)\]")

IN_ANGLE_BRACKETS_REGEX = regex.compile(r"<((?>[^<>]+|(?R))*)>")

IN_CURLY_BRACKETS_REGEX = regex.compile(r"{((?>[^{}]+|(?R))*)}")

CONTAINMENT_REGEXES = [
    ('in_quotes', IN_QUOTES_REGEX),
    ('in_parentheses', IN_PARENTHESES_REGEX),
    ('in_brackets', IN_BRACKETS_REGEX),
    ('in_angle_brackets', IN_ANGLE_BRACKETS_REGEX),
    ('in_curly_brackets', IN_CURLY_BRACKETS_REGEX)
]


class ContainmentClassificationFeatureDataset(FeatureDataset):
    def __init__(self, name, containment_regexes):
        self.name = name
        self.containment_regexes = containment_regexes

    def make(self, dataset_config, args, raw_dataset, tokenizer, cache=True):
        pass

    def prepare_dataset(self, exp_cfg):
        pass


# for containment regexes
def containment_mapping(token_list, tokenizer, containment_regexes):
    vocab = tokenizer.get_vocab()
    ix2str = {v: tokenizer.decode(v) for v in vocab.values()}
    token_length = {k: len(v) for k, v in ix2str.items()}

    get_tok_len = np.vectorize(lambda x: token_length[x])
    get_tok_str = np.vectorize(lambda x: ix2str[x])

    regex_matches = {name: [] for name, _ in containment_regexes}
    for i in range(len(token_list)):
        token_seq = token_list[i]
        decoded_seq = ''.join(get_tok_str(token_seq))
        tok_lens = get_tok_len(token_seq)
        sequence_indices = np.cumsum(tok_lens)

        char2tok = get_char_to_tok_map(decoded_seq, sequence_indices)

        for name, reg in containment_regexes:
            feature_char_spans = [m.span() for m in reg.finditer(decoded_seq)]
            valid_spans = [
                span for span in feature_char_spans if span[1] - span[0] > 2]
            token_spans = [(char2tok[start]+1, char2tok[end-1]-1)
                           for start, end in valid_spans]
            regex_matches[name].append(token_spans)
    return regex_matches
