import numpy as np
import os
import datasets
from scipy import sparse


class FeatureDataset:

    def __init__(self, name):
        self.name = name

    def load(self, dataset_config):
        file_loc = os.path.join(
            os.getenv('FEATURE_DATASET_DIR', 'feature_datasets'),
            dataset_config.make_dir_name()
        )
        return datasets.load_from_disk(file_loc)

    def save(self, dataset_config, feature_dataset):
        file_loc = os.path.join(
            os.getenv('FEATURE_DATASET_DIR', 'feature_datasets'),
            dataset_config.make_dir_name()
        )
        os.makedirs(file_loc, exist_ok=True)
        feature_dataset.save_to_disk(file_loc)


def get_char_to_tok_map(decoded_seq, sequence_indices):
    sequence_indices = np.concatenate([np.array([0]), sequence_indices])
    char_to_tok = np.zeros(len(decoded_seq), dtype=int)
    for i, (start, end) in enumerate(zip(sequence_indices[:-1], sequence_indices[1:])):
        char_to_tok[start:end] = i
    return char_to_tok


def tokenize_and_concatenate_separate_subsequences(
    dataset,
    tokenizer,
    streaming=False,
    max_length=1024,
    column_name="text",
    add_bos_token=True,
    random_subsequence=True,
    num_proc=10,
):
    """Helper function to process text
    Args:
        dataset (Dataset): The dataset to tokenize, assumed to be a HuggingFace text dataset.
        tokenizer (AutoTokenizer): The tokenizer. Assumed to have a bos_token_id and an eos_token_id.
        streaming (bool, optional): Whether the dataset is being streamed. If True, avoids using parallelism. Defaults to False.
        max_length (int, optional): The length of the context window of the sequence. Defaults to 1024.
        column_name (str, optional): The name of the text column in the dataset. Defaults to 'text'.
        add_bos_token (bool, optional): . Defaults to True.
    Returns:
        Dataset: Returns the tokenized dataset, as a dataset of tensors, with a single column called "tokens"
    """
    def subsample_sequence(tokens, max_length, pad_id, add_bos_token=True, random_subsequence=True):
        """Subsample a sequence to a maximum length, padding if necessary."""
        if add_bos_token:
            seq_len = max_length - 1
        else:
            seq_len = max_length

        tok_seqs = []
        for tok_seq in tokens:
            if len(tok_seq) > seq_len:
                # Take random seq_len length subsequence
                start = np.random.randint(
                    0, len(tok_seq) - seq_len) if random_subsequence else 0
                tok_seq = tok_seq[start: start + seq_len]
            else:
                # Pad to seq_len
                tok_seq = np.pad(
                    tok_seq, (0, seq_len - len(tok_seq)), constant_values=pad_id)
            tok_seqs.append(tok_seq)

        tok_arr = np.array(tok_seqs)
        if add_bos_token:
            bos_arr = np.ones((len(tok_seqs), 1),
                              dtype=np.int32) * tokenizer.bos_token_id
            tok_arr = np.hstack([bos_arr, tok_arr])

        return tok_arr.astype(np.int32).tolist()

    if tokenizer.pad_token is None:
        print('WARNING: model does not have a pad token')
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    # Define the length to chop things up into - leaving space for a bos_token if required

    tokenized_dataset = dataset.map(
        lambda x: {'all_tokens': tokenizer(x[column_name])['input_ids']},
        batched=True,
        num_proc=(num_proc if not streaming else None),
    )
    print('Finished tokenizing dataset, beginning to subsample sequences')
    subsampled_sequences = subsample_sequence(
        tokenized_dataset['all_tokens'], max_length, tokenizer.pad_token_id, add_bos_token, random_subsequence)
    tokenized_dataset = tokenized_dataset.add_column(
        'tokens', subsampled_sequences)

    tokenized_dataset.set_format(type="torch")

    return tokenized_dataset
