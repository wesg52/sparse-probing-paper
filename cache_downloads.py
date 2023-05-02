# Script for saving models and data to centralized storage on the cluster.
# This script only ever needs to be run one time per model or dataset.
# Use methods in [load.py] to access the downloaded models/datasets.

# Make sure environment variables are set before running this script

import argparse
import os


def save_model(model_name):
    # Just loading the model once will cache it to TRANSFORMERS_CACHE
    from transformer_lens import HookedTransformer
    HookedTransformer.from_pretrained(model_name, device='cpu')


def save_dataset(dataset_name, split):
    import datasets
    dataset = datasets.load_dataset(dataset_name, split=split)
    # cache doesn't work since there is no loader script
    # see workaround https://github.com/huggingface/datasets/issues/3547#issuecomment-1252503988
    save_path = os.path.join(
        os.environ['HF_DATASETS_CACHE'], dataset_name, split)
    dataset.save_to_disk(save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--model', default=None, help='Name of model from TransformerLens')
    parser.add_argument(
        '-d', '--dataset', default=None, help='Name of dataset from HF')
    parser.add_argument(
        '-s', '--split', default=None, help='Name of split for dataset from HF')

    args = vars(parser.parse_args())

    if args['model'] is not None:
        model = args['model']
        print(f'Saving model {model}')
        save_model(model)

    if args['dataset'] is not None:
        dataset = args['dataset']
        split = args['split'] if args['split'] is not None else 'train'
        print(f'Saving split {split} of {dataset}')
        save_dataset(dataset, split)
