import os

import torch
from datasets.arrow_dataset import Dataset
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformer_lens import HookedTransformer

from activations.common import (get_hook_name, load_json, load_tensor,
                                save_json, save_tensor)


@torch.no_grad()
def get_full_activation_tensor(
    model: HookedTransformer,
    dataset: Dataset,
    device=None,
    batch_size=16,
    save_postactivation=False,
    verbose=False,
    layers=None,
    positions=None,
    flatten_and_ignore_padding=False
):
    '''
    Collect activations for all examples in a dataset

    Returns a (n_layer, n_neuron, n_sequence, seq_len) tensor of activations
    '''
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if layers is None:
        layers = list(range(model.cfg.n_layers))
    if positions is None:
        positions = list(range(len(dataset[0]['tokens'])))

    # make hooks
    layer_names = [get_hook_name(lix, save_postactivation) for lix in layers]
    layer_ix = {name: lix for lix, name in enumerate(layer_names)}

    # layer x sequence_dim x position x neuron
    n_seqs, ctx_len = dataset['tokens'].shape
    activation_shape = (len(layers), n_seqs, len(positions), model.cfg.d_mlp)
    activations = torch.zeros(activation_shape, dtype=torch.float16)

    batch_num = 0  # nonlocal variable to include in hooks

    def save_hook(tensor, hook):
        nonlocal batch_num
        nonlocal batch_size
        offset = batch_num * batch_size
        layer = layer_ix[hook.name]
        batch_act = tensor.detach().cpu()[:, positions, :].to(torch.float16)
        activations[layer, offset:offset + batch_size, :, :] = batch_act

    for name in layer_names:
        model.add_hook(name, save_hook)

    # iterate over dataset
    # TODO: can this loop be put into its own function and reused across the activations experiments?
    dataloader = DataLoader(
        dataset['tokens'], batch_size=batch_size, shuffle=False)
    for batch in tqdm(dataloader, disable=not verbose):
        model.forward(batch.to(device), return_type=None)
        batch_num += 1
    model.reset_hooks()

    # to layer, neuron, sequence, position
    activations = rearrange(activations, 'l s p n -> l n s p')
    if flatten_and_ignore_padding:
        # layer, neuron, (sequence, position)
        activations = activations[:, :, dataset['tokens'] > 1]

    return activations


def save_full_activation_tensor(experiment_dir, activations, metadata, output_precision=16):
    '''
    Save activations and metadata to disk
    '''
    os.makedirs(experiment_dir, exist_ok=True)
    # TODO(wesg): add 8bit option
    dtype = torch.float32 if output_precision == 32 else torch.float16
    save_tensor(os.path.join(experiment_dir, 'activations.pt'),
                activations.to(dtype))
    save_json(os.path.join(experiment_dir, 'metadata.json'), metadata)


def load_activation_all(experiment_dir):
    '''
    Load activations and metadata from disk
    '''
    activations = load_tensor(os.path.join(experiment_dir, 'activations.pt'))
    metadata = load_json(os.path.join(experiment_dir, 'metadata.json'))
    return activations, metadata
