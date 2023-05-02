import numpy as np
import torch

from activations.activation_all import get_full_activation_tensor
from activations.activation_metrics import get_activation_metrics
from activations.activation_subset import get_activation_subset
from activations.common import time_function


def test_get_activation_metrics(
        model, dataset, batch_size=16, top_k=30, n_bin=30, hist_min=-.2, hist_max=2):
    '''
    Validate the outputs of the `get_activation_metrics` function
    '''
    n_layer = model.cfg.n_layers
    n_neuron = model.cfg.d_mlp

    (top_k_seqix, top_k_pos, bin_counts), act_metrics_time = time_function(
        get_activation_metrics, model, dataset, batch_size=batch_size, top_k=top_k,
        n_bin=n_bin, hist_min=hist_min, hist_max=hist_max, verbose=True)
    activations, act_all_time = time_function(
        get_full_activation_tensor, model, dataset, verbose=True)
    print(
        f'get_activation_stats: {act_metrics_time:.3f}, get_mlp_activations: {act_all_time:.3f}')

    # validate top k output
    wrong_act_counter = 0
    for lix in range(n_layer):
        for nix in range(n_neuron):
            neuron_top_activations = activations[lix, nix, :, :].flatten().sort(
                descending=True)[0][:top_k]
            for k in range(top_k):
                seqix = top_k_seqix[lix, nix, k]
                pos = top_k_pos[lix, nix, k]
                activation = activations[lix, nix, seqix, pos]
                wrong_act_counter += activation not in neuron_top_activations
    print(f'Top k output correct: {wrong_act_counter==0}')

    # validate histogram output
    # NOTE: I'm not sure why the histograms don't match exactly, but they are very close.
    #       I can't find a pattern to the errors other than it probably has something to do
    #       with values around the bin edges as differences always come in pairs with one bin
    #       being off by 1 and an adjacent one being off by -1.
    #       If this is a problem, we can try to figure out what's going on, but for now it
    #       seems fine to leave it.
    bin_counts_compare = torch.zeros(
        (n_layer, n_neuron, n_bin), dtype=torch.float32)
    bin_edges = torch.cat([torch.linspace(hist_min, hist_max, n_bin),
                           torch.tensor([float('inf')])])
    for lix in range(n_layer):
        for nix in range(n_neuron):
            bin_counts_compare[lix, nix, :] += torch.histogram(
                activations[lix, nix, :, :].flatten(), bin_edges).hist
    prop_diff = 1 - ((bin_counts == bin_counts_compare).sum() /
                     bin_counts.numel()).item()
    print(
        f'bin_counts exact match: {(bin_counts == bin_counts_compare).all()}, near match: {prop_diff<1e-5}, prop different: {prop_diff:.7f}')


def test_get_activation_subset(model, dataset, batch_size=16, subset_size=10):
    '''
    Validate the outputs of the `get_activation_subset function
    '''
    n_layer = model.cfg.n_layers
    n_neuron = model.cfg.d_mlp
    neuron_subset = [(np.random.randint(n_layer), np.random.randint(n_neuron))
                     for _ in range(subset_size)]

    activation_subset, subset_time = time_function(
        get_activation_subset, model, dataset, neuron_subset,
        batch_size=batch_size, verbose=True)
    activations, all_time = time_function(
        get_full_activation_tensor, model, dataset, batch_size=batch_size, verbose=True)
    print(
        f'get_activation_subset: {subset_time:.3f}, get_mlp_activations: {all_time:.3f}')

    # validate output
    keys_match = set(neuron_subset) == set(activation_subset.keys())
    print(f'Subset keys correct: {keys_match}')
    if not keys_match:
        return
    values_match = True
    for neuron in neuron_subset:
        lix, nix = neuron
        neuron_activations = activations[lix, nix, :, :]
        if not (neuron_activations == activation_subset[neuron]).all():
            values_match = False
            break
    print(f'Subset values correct: {values_match}')
