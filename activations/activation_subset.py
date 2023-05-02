import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformer_lens import HookedTransformer
from functools import partial
from concurrent.futures import ProcessPoolExecutor

from activations.common import (get_hook_name, load_json, load_tensor,
                                save_json, save_tensor)


@torch.no_grad()
def get_activation_subset(
    model: HookedTransformer,
    dataset,
    neuron_subset,  # list[tuple[int, int]],
    batch_size=16,
    save_postactivation=False,
    device=None,
    verbose=False,
):
    '''
    Find the activations over the dataset for each neuron in the provided subset.

    Arguments
    model: HookedTransformer model
    dataset: unbatched text dataset
    neuron_subset: list of (layer_index, neuron_index) tuples
    batch_size: batch size for the data loader
    device: "cpu" or "cuda," inferred automatically if None
    verbose: whether to print progress bar

    Returns
    activation_subset: dictionary containing a (n_seq, n_ctx) tensor of activations for each neuron in the subset
    '''
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # add hooks
    hook_names = [get_hook_name(lix, save_postactivation)
                  for lix in range(model.cfg.n_layers)]
    unique_layers = set([lix for lix, _ in neuron_subset])
    layer_neuron_subsets = {hook_names[lix1]: [
        nix for lix2, nix in neuron_subset if lix1 == lix2] for lix1 in unique_layers}
    layer_subset_activations = {hook_names[lix]: [] for lix in unique_layers}

    def save_subset_hook(tensor, hook):
        layer_subset_activations[hook.name].append(
            tensor.detach().cpu()[:, :, layer_neuron_subsets[hook.name]])

    for lix in unique_layers:
        model.add_hook(hook_names[lix], save_subset_hook)

    # iterate over dataset
    dataloader = DataLoader(
        dataset['tokens'], batch_size=batch_size, shuffle=False)
    for batch in tqdm(dataloader, disable=not verbose):
        model.forward(
            batch.to(device),
            return_type=None,
            stop_at_layer=max(unique_layers) + 1)
    model.reset_hooks()

    # aggregate
    activation_subset = {}
    for lix in unique_layers:
        layer_activations = torch.cat(
            layer_subset_activations[hook_names[lix]], dim=0)
        for i, nix in enumerate(layer_neuron_subsets[hook_names[lix]]):
            activation_subset[(lix, nix)] = layer_activations[:, :, i]

    return activation_subset


def compute_vocab_summary_df(item, dataset, decoded_vocab):
    (l, n), activations = item
    activation_df = pd.DataFrame({
        'token': dataset['tokens'].flatten(),
        'activation': activations.numpy().flatten().astype(np.float32),
    })
    token_summary_df = activation_df.groupby('token')['activation'].describe()

    token_summary_df['string'] = [
        decoded_vocab[t] for t in token_summary_df.index]
    return (l, n), token_summary_df


def parallelize_activation_processing(activation_subset, dataset, decoded_vocab):
    neuron_token_summary_dfs = {}
    n_workers = int(os.getenv('SLURM_CPUS_PER_TASK', 10))
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        process_partial = partial(
            compute_vocab_summary_df, dataset=dataset, decoded_vocab=decoded_vocab)
        results = executor.map(process_partial, activation_subset.items())

    for key, token_summary_df in results:
        neuron_token_summary_dfs[key] = token_summary_df

    return neuron_token_summary_dfs


def compute_token_summary_dfs_parallel(activation_subset, dataset, tokenizer):
    '''
    Compute summary statistics for each token in the dataset for each neuron in the subset

    Does not work on slurm.
    '''
    decoded_vocab = {t: tokenizer.decode(t)
                     for t in tokenizer.get_vocab().values()}
    neuron_token_summary_dfs = {}

    neuron_token_summary_dfs = parallelize_activation_processing(
        activation_subset, dataset, decoded_vocab)
    token_summary_df = pd.concat(neuron_token_summary_dfs)
    token_summary_df.index.names = ['layer', 'neuron', 'token']

    return token_summary_df.reset_index()


def compute_token_summary_dfs(activation_subset, dataset, tokenizer):
    '''
    Compute summary statistics for each token in the dataset for each neuron in the subset
    '''
    decoded_vocab = {t: tokenizer.decode(t)
                     for t in tokenizer.get_vocab().values()}
    neuron_token_summary_dfs = {}
    for (l, n), activations in activation_subset.items():
        activation_df = pd.DataFrame({
            'token': dataset['tokens'].flatten(),
            'activation': activations.numpy().flatten().astype(np.float32),
        })
        token_summary_df = activation_df.groupby(
            'token')['activation'].describe()
        token_summary_df['string'] = [
            decoded_vocab[t] for t in token_summary_df.index]
        neuron_token_summary_dfs[(l, n)] = token_summary_df

    token_summary_df = pd.concat(neuron_token_summary_dfs)
    token_summary_df.index.names = ['layer', 'neuron', 'token']

    return token_summary_df.reset_index()


def adjust_precision(activation_tensor, output_precision=16):
    '''
    Adjust the precision of the activation subset
    '''
    if output_precision == 32:
        return activation_tensor.to(torch.float32)
    elif output_precision == 16:
        return activation_tensor.to(torch.float16)
    elif output_precision == 8:
        min_val = activation_tensor.min().item()
        max_val = activation_tensor.max().item()
        num_quant_levels = 2**output_precision
        scale = (max_val - min_val) / (num_quant_levels - 1)
        zero_point = round(-min_val / scale)
        return torch.quantize_per_tensor(activation_tensor, scale, zero_point, torch.quint8)
    else:
        raise ValueError(f'Invalid output precision: {output_precision}')


def save_activation_subset(
        experiment_dir, activation_subset, token_summary_df, metadata,
        output_precision=16, save_by_neuron=False):
    '''
    Save the activation subset and metadata
    '''
    os.makedirs(experiment_dir, exist_ok=True)

    activation_subset = {
        k: adjust_precision(v, output_precision)
        for k, v in activation_subset.items()
    }
    if save_by_neuron:
        for k, v in activation_subset.items():
            save_tensor(os.path.join(experiment_dir,
                        f'{k[0]}.{k[1]}.pt'), v)
    else:
        save_tensor(os.path.join(experiment_dir,
                    'activation_subset_dict.pt'), activation_subset)

    save_json(os.path.join(experiment_dir, 'metadata.json'), metadata)
    if token_summary_df is not None:
        df_path = os.path.join(experiment_dir, 'token_summary_df.csv')
        token_summary_df.to_csv(df_path, index=False)


def load_activation_subset(
        model_name, dataset_name, experiment_name,
        results_dir='results', experiment_type='activation_subset', layer=None, neuron=None):
    '''
    Load the activation subset and metadata
    '''
    experiment_path = os.path.join(
        results_dir, experiment_type, model_name, dataset_name, experiment_name)

    if layer is not None and neuron is not None:
        activation_subset = load_tensor(os.path.join(
            experiment_path, f'{layer}.{neuron}.pt'))
    else:
        activation_subset = load_tensor(os.path.join(
            experiment_path, 'activation_subset_dict.pt'))

    metadata = load_json(os.path.join(experiment_path, 'metadata.json'))
    if 'token_summary_df.csv' in set(os.listdir(experiment_path)):
        token_summary_df = pd.read_csv(os.path.join(
            experiment_path, 'token_summary_df.csv'))
        return activation_subset, metadata, token_summary_df
    else:
        return activation_subset, metadata


def save_neuron_subset(neuron_subset, path):
    '''
    Save a neuron subset of the form [(lix, nix), ...] to a file at `path`
    '''
    save_tensor(path, neuron_subset)


def load_neuron_subset(path):
    '''
    Load a neuron subset of the form [(lix, nix), ...] to a file at `path`
    '''
    return load_tensor(path)


def save_neuron_subset_csv(df, model_name, subset_name):
    '''
    Save a neuron subset to csv file
    '''
    assert 'layer' in df.columns and 'neuron' in df.columns
    path = os.path.join(
        os.environ.get('INTERPRETABLE_NEURONS_DIR', 'interpretable_neurons'),
        model_name
    )
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, subset_name + '.csv')
    df.to_csv(file_path, index=False)


def load_neuron_subset_csv(args, return_df=False):
    '''
    Load a neuron subset of the form [(lix, nix), ...] from csv file
    '''
    path = os.path.join(
        os.environ.get('INTERPRETABLE_NEURONS_DIR', 'interpretable_neurons'),
        args.model,
        args.neuron_subset_file
    )
    ndf = pd.read_csv(path)

    if args.auto_restrict_neuron_subset_file:
        fds = args.feature_dataset.split('.')[0]
        neuron_ds = np.array(ndf.dataset.apply(lambda x: x.split('.')[0]))
        ndf = ndf.loc[neuron_ds == fds]

    neuron_subset = [
        (int(lix), int(nix)) for lix, nix
        in zip(ndf['layer'].values, ndf['neuron'].values)
    ]
    if return_df:
        return neuron_subset, ndf
    else:
        return neuron_subset


def parse_neuron_str(neuron_str: str):
    lix, nix = neuron_str.split(',')
    return (int(lix), int(nix))
