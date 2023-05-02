from .common import *
from transformer_lens import utils
import numpy as np
import torch
from datasets import Dataset

# token_ix: [list of prefixes to create classes for]
# each list containts a set of prefixes
# where a prefix is a tuple of tokens
PYTHIA_70M_L1_N111_STIMULI = {
    12299: [[(13804,), (9432,)]],  # e.g. {Har, har}|vard
    35476: [[(43950, 762, 253), (3567, 762, 253)], [(2256, 281, 253)]],  # Boost
    20740: [[(2058, 253, 5403), (3404, 253, 5403)]],  # census
    26268: [[(15, 12332, 9824)], [(32170, 9824)]],  # Chain
    412: [[(47678, ), (14029, )]],  # op
    20000: [[(38476, 5625)]],  # Peace
    3621: [[(47694, )]],  # lease
    13606: [[(5625, 330)]],  # oven
    14894: [[(12602, 273, 15123)]],  # hma
    2616: [[(4335, )], [(29602, )]],  # factors
    7736: [[(21698, )]],  # pogenic
    39098: [[(4146, 12761), (14594, 12761)]],  # systems
    4412: [[(11586, 2077)]],  # District
    15353: [[(36642, )]],  # gate
    11845: [[(19256, 38056)]],  # iallance
    4694: [[(47678, ), (14029, )]],  # vision
    20310: [[(5625, 10518), (22817, 10518)]],  # Mach
    6875: [[(7671, ), (31351, )]],  # Science
    17629: [[(6399, )]],  # ograms
    19934: [[(35654, 15)], [(2700, 15)]],  # apple
    48862: [[(45590, 64)]],  # AUX
    7404: [[(749, ), (2377, )]],  # process
    35437: [[(22817, 13940), (5625, 13940)]],  # communication
    7662: [[(2359, 412), (22468, 412)]],  # roduction
    25837: [[(681, 16)]],  # fw
    16240: [[(21034, )], [(20709, )]],  # ington
}


class NeuronStimulusFeatureDataset(FeatureDataset):
    def __init__(self, name, stimuli):
        self.name = name
        self.stimuli = stimuli

    def make(self, dataset_config, args, raw_dataset, tokenizer, cache=True):
        if 'tokens' in raw_dataset.column_names:  # already tokenized
            token_vector = raw_dataset['tokens'].flatten().numpy()
        else:
            tokenized_ds = utils.tokenize_and_concatenate(
                raw_dataset, tokenizer)
            token_vector = tokenized_ds['tokens'].flatten().numpy()

        stimulus_datasets = []
        for probe_token, stimulus_class in self.stimuli.items():
            # just take first class for simplicity (should be the strongest activating)
            stimulus = stimulus_class[0]

            probe_token_indices = np.where(token_vector == probe_token)[0]

            valid_stimulus_indices = []
            for ngram_prefix in stimulus:
                probe_indices_with_stimulus_prefix = probe_token_indices
                for ix, t in enumerate(ngram_prefix[::-1]):
                    offset = ix + 1
                    probe_indices_with_correct_stimuli_prefix = np.where(
                        token_vector[probe_indices_with_stimulus_prefix - offset] == t)[0]
                    probe_indices_with_stimulus_prefix = probe_indices_with_stimulus_prefix[
                        probe_indices_with_correct_stimuli_prefix]
                valid_stimulus_indices.append(
                    probe_indices_with_stimulus_prefix)
            valid_stimulus_indices = np.concatenate(valid_stimulus_indices)
            valid_negative_stimulus_indices = np.setdiff1d(
                probe_token_indices, valid_stimulus_indices)

            target_n_positive = min(500, len(valid_stimulus_indices))
            target_n_negative = min(2000, len(valid_negative_stimulus_indices))
            ctx_len = 32

            positive_indices = np.sort(np.random.choice(
                valid_stimulus_indices, target_n_positive, replace=False))
            negative_indices = np.sort(np.random.choice(
                valid_negative_stimulus_indices, target_n_negative, replace=False))
            len(positive_indices), len(negative_indices)

            positive_stimulus_token_tensor = np.vstack([
                token_vector[ix+1-ctx_len: ix+1] for ix in positive_indices
            ])
            negative_stimulus_token_tensor = np.vstack([
                token_vector[ix+1-ctx_len: ix+1] for ix in negative_indices
            ])
            stimulus_token_tensor = np.vstack([
                positive_stimulus_token_tensor, negative_stimulus_token_tensor
            ])

            token_name = tokenizer.decode(probe_token)
            feature_prefix = tokenizer.decode(list(stimulus[0]))
            feature_name = f'{feature_prefix}|{token_name}|'
            labels = ['positive' for _ in range(len(positive_indices))] \
                + ['negative' for _ in range(len(negative_indices))]

            stimulus_ds = datasets.Dataset.from_dict({
                'tokens': stimulus_token_tensor,
                'label': labels,
                'feature_name': [feature_name for _ in range(len(labels))]
            }).shuffle()
            stimulus_datasets.append(stimulus_ds)
            print(
                f'Finished token {probe_token} {feature_name} with {len(positive_indices)} positive and {len(negative_indices)} negative stimuli')

        neuron_stimulus_dataset = datasets.concatenate_datasets(
            stimulus_datasets)
        neuron_stimulus_dataset.set_format(type="torch")

        target_n_positive = min(500, len(valid_stimulus_indices))
        target_n_negative = min(2000, len(valid_negative_stimulus_indices))
        ctx_len = args.get('ctx_len', 32)

        if cache:
            self.save(dataset_config, neuron_stimulus_dataset)

        return neuron_stimulus_dataset

    def prepare_dataset(self, exp_cfg):
        raise NotImplementedError(
            'Currently neuron stimulus is only for activations')
