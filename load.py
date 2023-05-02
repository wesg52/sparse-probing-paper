import os
import torch
import pickle
import datasets
from transformer_lens import HookedTransformer, utils
from config import FeatureDatasetConfig


def load_model(model_name="gpt2-small", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_name.startswith('pythia'):
        try:
            model = HookedTransformer.from_pretrained(
                model_name + '-70m', device='cpu')
        except ValueError:
            print(f'No {model_name}-v0 available')
    model = HookedTransformer.from_pretrained(model_name, device='cpu')
    model.eval()
    torch.set_grad_enabled(False)
    if model.cfg.device != device:
        try:
            model.to(device)
        except RuntimeError:
            print(
                f"WARNING: model is too large to fit on {device}. Falling back to CPU")
            model.to('cpu')

    return model


def load_feature_dataset(name, n=-1):
    path = os.path.join(os.environ.get(
        'FEATURE_DATASET_DIR', 'feature_datasets'), name)
    if n > 0:
        return datasets.load_from_disk(path).select(range(n))
    else:
        return datasets.load_from_disk(path)


def load_raw_dataset(path, n_seqs=-1):
    save_path = os.path.join(os.environ['HF_DATASETS_CACHE'], path)
    dataset = datasets.load_from_disk(save_path)
    if n_seqs > 0:
        dataset = dataset.select(range(n_seqs))
    return dataset
