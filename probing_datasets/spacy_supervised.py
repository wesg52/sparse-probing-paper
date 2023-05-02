import torch
import spacy
import numpy as np

import spacy_alignments as tokenizations

# Tokenization info
# Part of speech, Tag, DEP
# Is stop?
# Something with the parse tree? Eg, n_lefts>0 or n_rights>0
# Named entities
# end of sentence
# see https://spacy.io/api/token for others

# Morphology see https://universaldependencies.org/u/feat/index.html
# - Number
# - Gender?
# - Person
# - Prontype
# - Tense
# - Verbform
# - Aspect
# - Degree?
# - Numtype


def make_spacy_supervised_dataset(model, dataset, n=200):
    nlp = spacy.load("en_core_web_sm")

    seq_len = len(dataset[0]['tokens'])

    pos_matrix = np.zeros((n, seq_len), dtype='<U5')
    tag_matrix = np.zeros((n, seq_len), dtype='<U5')
    dep_matrix = np.zeros((n, seq_len), dtype='<U10')
    ent_type_matrix = np.zeros((n, seq_len), dtype='<U8')

    morph_number_matrix = np.zeros((n, seq_len), dtype='<U5')
    morph_person_matrix = np.zeros((n, seq_len), dtype='<U5')
    morph_prontype_matrix = np.zeros((n, seq_len), dtype='<U5')
    morph_tense_matrix = np.zeros((n, seq_len), dtype='<U5')
    morph_verbform_matrix = np.zeros((n, seq_len), dtype='<U5')
    morph_aspect_matrix = np.zeros((n, seq_len), dtype='<U5')

    is_alpha_matrix = np.zeros((n, seq_len), dtype='<U5')
    is_stop_matrix = np.zeros((n, seq_len), dtype='<U5')
    is_sent_end_matrix = np.zeros((n, seq_len), dtype='<U5')
    is_sent_begins_matrix = np.zeros((n, seq_len), dtype='<U5')

    for i in range(n):
        token_ixs = dataset['tokens'][i][1:]

        seq = model.to_string(token_ixs)
        doc = nlp(seq)

        model_tokens = model.to_str_tokens(token_ixs)
        spacy_tokens = [t.text for t in doc]

        spacy_pos = np.array([t.pos_ for t in doc])
        spacy_tag = np.array([t.tag_ for t in doc])
        spacy_dep = np.array([t.dep_ for t in doc])
        spacy_ent_type = np.array([t.ent_type_ for t in doc])

        spacy_morph_number = np.array(
            [t.morph.to_dict().get('Number', '') for t in doc])
        spacy_morph_person = np.array(
            [t.morph.to_dict().get('Person', '') for t in doc])
        spacy_morph_prontype = np.array(
            [t.morph.to_dict().get('PronType', '') for t in doc])
        spacy_morph_tense = np.array(
            [t.morph.to_dict().get('Tense', '') for t in doc])
        spacy_morph_verbform = np.array(
            [t.morph.to_dict().get('VerbForm', '') for t in doc])
        spacy_morph_aspect = np.array(
            [t.morph.to_dict().get('Aspect', '') for t in doc])

        spacy_is_alpha = np.array([t.is_alpha for t in doc])
        spacy_is_stop = np.array([t.is_stop for t in doc])

        spacy_end_sent = np.array([t.is_sent_end for t in doc])
        spacy_begin_sent = np.array([t.is_sent_start for t in doc])

        gpt2spacy, spacy2gpt = tokenizations.get_alignments(
            model_tokens, spacy_tokens)

        # offset by 1 for BOS
        pos_matrix[i, 1:] = get_model_labels(gpt2spacy, spacy_pos)
        tag_matrix[i, 1:] = get_model_labels(gpt2spacy, spacy_tag)
        dep_matrix[i, 1:] = get_model_labels(gpt2spacy, spacy_dep)
        ent_type_matrix[i, 1:] = get_model_labels(gpt2spacy, spacy_ent_type)

        morph_number_matrix[i, 1:] = get_model_labels(
            gpt2spacy, spacy_morph_number)
        morph_person_matrix[i, 1:] = get_model_labels(
            gpt2spacy, spacy_morph_person)
        morph_prontype_matrix[i, 1:] = get_model_labels(
            gpt2spacy, spacy_morph_prontype)
        morph_tense_matrix[i, 1:] = get_model_labels(
            gpt2spacy, spacy_morph_tense)
        morph_verbform_matrix[i, 1:] = get_model_labels(
            gpt2spacy, spacy_morph_verbform)
        morph_aspect_matrix[i, 1:] = get_model_labels(
            gpt2spacy, spacy_morph_aspect)

        is_alpha_matrix[i, 1:] = get_model_labels(gpt2spacy, spacy_is_alpha)
        is_stop_matrix[i, 1:] = get_model_labels(gpt2spacy, spacy_is_stop)
        # spacy always deleclares end of sequence true
        is_sent_end_matrix[i, 1:-1] = \
            get_model_labels(gpt2spacy, spacy_end_sent)[:-1]
        is_sent_begins_matrix[i, 1:] = get_model_labels(
            gpt2spacy, spacy_begin_sent)

    # see https://universaldependencies.org/u/pos/
    pos_labels = {
        'is_adj': pos_matrix == 'ADJ',
        'is_adp': pos_matrix == 'ADP',
        'is_det': pos_matrix == 'DET',
        'is_noun': pos_matrix == 'NOUN',
        'is_pron': pos_matrix == 'PRON',
        'is_propn': pos_matrix == 'PROPN',
        'is_punct': pos_matrix == 'PUNCT',
        'is_verb': pos_matrix == 'VERB',
    }
    dependency_labels = {
        'is_root': dep_matrix == 'ROOT',
        'is_advmod': dep_matrix == 'advmod',
        'is_amod': dep_matrix == 'amod',
        'is_compound': dep_matrix == 'compound',
        'is_conj': dep_matrix == 'conj',
        'is_dep': dep_matrix == 'dep',
        # 'is_det': dep_matrix == 'det',
        'is_dobj': dep_matrix == 'dobj',
        'is_pobj': dep_matrix == 'pobj',
        'is_prep': dep_matrix == 'prep',
    }

    morph_labels = {
        'is_singular': morph_number_matrix == 'Sing',
        'is_plural': morph_number_matrix == 'Plur',

        'is_third_person': morph_person_matrix == '3',
        'is_first_person': morph_person_matrix == '1',

        'is_prs_pron': morph_prontype_matrix == 'Prs',
        'is_art_pron': morph_prontype_matrix == 'Art',

        'is_past_tense': morph_tense_matrix == 'Past',
        'is_present_tense': morph_tense_matrix == 'Pres',

        'is_fin_verb': morph_verbform_matrix == 'Fin',
        'is_part_verb': morph_verbform_matrix == 'Part',
        'is_inf_verb': morph_verbform_matrix == 'Inf',

        'is_perf_aspect': morph_aspect_matrix == 'Perf',
        'is_prog_aspect': morph_aspect_matrix == 'Prog',
    }

    ent_labels = {
        'is_org_ent': ent_type_matrix == 'ORG',
        'is_person_ent': ent_type_matrix == 'PERSON'
    }

    misc_labels = {
        'is_stop': is_stop_matrix == 'True',
        'sent_end': is_sent_end_matrix == 'True',
        'sent_begin': is_sent_begins_matrix == 'True',
    }

    label_sets = [
        pos_labels, dependency_labels, morph_labels, ent_labels, misc_labels
    ]
    full_label_dict = {k: v for label_set in label_sets
                       for k, v in label_set.items()}
    return full_label_dict


def get_spacy_tag(tok_ix2spacy_ixs, spacy_attribute_list):
    if len(tok_ix2spacy_ixs) == 1:
        return spacy_attribute_list[tok_ix2spacy_ixs[0]]
    elif len(tok_ix2spacy_ixs) == 0:
        return 'NONE'
    else:
        return 'MULTI'


def get_model_labels(model2spacy, spacy_tag):
    return np.array([
        get_spacy_tag(tok_ix2spacy_ixs, spacy_tag)
        for tok_ix2spacy_ixs in model2spacy
    ])
