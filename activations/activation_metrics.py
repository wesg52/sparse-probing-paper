import os

import torch
from datasets.arrow_dataset import Dataset
from einops import rearrange, repeat
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial
from functorch import vmap
from transformer_lens import HookedTransformer

from activations.common import (
    get_hook_name, load_json, load_tensor,
    save_json, save_tensor)


@torch.no_grad()
def get_activations_hist(activations, hist_min, hist_max, n_bin):
    '''
    Compute independent histograms for each neuron in the activations tensor.
    '''
    n_layer, n_neuron, _ = activations.shape

    # layer, neuron, activations (= batch_size * seq_len)
    all_activations_by_neuron = rearrange(
        activations, 'l n a -> (l n) a')
    clamped_activations = torch.clamp(
        all_activations_by_neuron, hist_min + 1e-6, hist_max - 1e-6
    )

    bin_edges = torch.linspace(hist_min, hist_max, n_bin+1)
    binned_histogram = partial(torch.histogram, bins=bin_edges)
    vectorized_histogram = vmap(binned_histogram)
    hist_by_neuron, _ = vectorized_histogram(clamped_activations)

    # layer, neuron, bin
    return rearrange(
        hist_by_neuron.to(torch.long),
        '(l n) b -> l n b',
        l=n_layer, n=n_neuron, b=n_bin
    )


@torch.no_grad()
def get_activations_top_k(activations, top_k_values, top_k_indices, step, batch_size, seq_len):
    '''
    Compute top k most activating examples for each neuron over both the current
    batch and previous batches. Modifies top_k_values and top_k_indices in place.
    '''
    # TODO: collect top k for every bin in the histogram
    n_layer, n_neuron, top_k = top_k_values.shape
    cur_batch_size = activations.shape[2] / seq_len

    all_values = torch.cat([top_k_values, activations], dim=2)
    batch_indices = repeat(
        int(step*batch_size*seq_len) +
        torch.arange(int(seq_len*cur_batch_size)),
        'x -> l n x', l=n_layer, n=n_neuron)
    all_indices = torch.concat([top_k_indices, batch_indices], dim=2)
    new_indices = torch.empty((n_layer, n_neuron, top_k), dtype=torch.long)

    torch.topk(all_values, top_k, dim=2, sorted=True,
               out=(top_k_values, new_indices))
    torch.gather(all_indices, 2, new_indices, out=top_k_indices)


@torch.no_grad()
def get_activation_metrics(
    model: HookedTransformer,
    dataset: Dataset,
    device=None,
    batch_size=16,
    top_k=30,
    n_bin=100,
    hist_min=-10,
    hist_max=10,
    save_postactivation=False,
    verbose=False,
):
    '''
    For each neuron find the top k most activating examples and activation histogram over the provided dataset.

    Returns
    top_k_seqix: (n_layer, n_neuron, top_k) tensor of sequence index for top activating examples
    top_k_pos: (n_layer, n_neuron, top_k) tensor of position for top activating examples
    bin_counts: (n_layer, n_neuron, n_bin) tensor of histogram bin counts where the final bin captures values greater than hist_max
    '''
    n_layer = model.cfg.n_layers
    n_neuron = model.cfg.d_mlp
    seq_len = len(dataset[0]['tokens'])

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    bin_counts = torch.zeros((n_layer, n_neuron, n_bin), dtype=torch.long)
    top_k_values = torch.full(
        (n_layer, n_neuron, top_k), float('-inf'), dtype=torch.float32)
    top_k_indices = torch.empty((n_layer, n_neuron, top_k), dtype=torch.long)

    # add hooks
    mlp_activations = {}

    def save_hook(tensor, hook):
        mlp_activations[hook.name] = tensor.detach().cpu()

    layer_names = [get_hook_name(lix, save_postactivation)
                   for lix in range(n_layer)]
    for name in layer_names:
        model.add_hook(name, save_hook)

    # iterate over dataset
    dataloader = DataLoader(
        dataset['tokens'], batch_size=batch_size, shuffle=False)
    for step, batch in enumerate(tqdm(dataloader, disable=not verbose)):
        model.forward(batch.to(device), return_type=None)
        activations = rearrange([mlp_activations[name] for name in layer_names],
                                'layer seq token neuron -> layer neuron (seq token)')

        # update running top k and histogram
        get_activations_top_k(activations, top_k_values,
                              top_k_indices, step, batch_size, seq_len)
        bin_counts += get_activations_hist(activations,
                                           hist_min, hist_max, n_bin)

    model.reset_hooks()

    top_k_seqix = top_k_indices // seq_len
    top_k_pos = top_k_indices % seq_len
    return top_k_seqix, top_k_pos, bin_counts


def save_activation_metrics(experiment_dir, top_k_seqix, top_k_pos, bin_counts,
                            metadata, output_precision=16):
    '''
    Save the activation metrics results and metadata
    '''
    os.makedirs(experiment_dir, exist_ok=True)
    # TODO(wesg): precision
    save_tensor(
        os.path.join(experiment_dir, 'top_k_seqix.pt'), top_k_seqix)
    save_tensor(
        os.path.join(experiment_dir, 'top_k_pos.pt'), top_k_pos)
    save_tensor(
        os.path.join(experiment_dir, 'bin_counts.pt'), bin_counts)

    save_json(os.path.join(experiment_dir, 'metadata.json'), metadata)


def load_activation_metrics(experiment_dir):
    '''
    Load the activation metrics resutls and metadata
    '''
    top_k_seqix = load_tensor(os.path.join(experiment_dir, 'top_k_seqix.pt'))
    top_k_pos = load_tensor(os.path.join(experiment_dir, 'top_k_pos.pt'))
    bin_counts = load_tensor(os.path.join(experiment_dir, 'bin_counts.pt'))
    metadata = load_json(os.path.join(experiment_dir, 'metadata.json'))
    return top_k_seqix, top_k_pos, bin_counts, metadata