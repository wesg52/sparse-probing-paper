#!/bin/bash

python make_feature_datasets.py \
    --feature_collection programming_lang_id

python make_feature_datasets.py \
    --feature_collection natural_lang_id \
    --ignore_first_k 1 \
    --lang_id_n_tokens 2

python make_feature_datasets.py \
    --feature_collection distribution_id

python make_feature_datasets.py \
    --feature_collection position \
    --n_seqs 10000 \
    --seq_len 1024

python make_feature_datasets.py \
    --feature_collection text_features \
    --seq_len 256 \
    --n_seqs 10000

python make_feature_datasets.py \
    --feature_collection ewt

python make_feature_datasets.py \
    --feature_collection compound_words \
    --seq_len 24

python make_feature_datasets.py \
    --feature_collection latex \
    --seq_len 1024
