import os
import time
import torch
from config import *
from utils import *
from load import load_model, load_feature_dataset
from activations.activation_probing_dataset import get_activation_dataset, make_index_mask
from activations.activation_all import (
    get_full_activation_tensor, save_full_activation_tensor)
from activations.activation_subset import (
    parse_neuron_str, get_activation_subset,
    save_activation_subset, load_neuron_subset_csv,
    compute_token_summary_dfs)
from activations.activation_metrics import (
    get_activation_metrics, save_activation_metrics)
from activations.common import (
    get_experiment_dir, get_experiment_metadata, get_experiment_info_str)
from activations.test import (
    test_get_activation_subset, test_get_activation_metrics)
from make_feature_datasets import prepare_feature_dataset


def make_activation_probe_dataset(args, model):
    '''
    Make the activation probe dataset
    '''
    args = vars(parser.parse_args())
    feature_dataset_cfg = parse_dataset_args(args['feature_dataset'])
    exp_cfg = ExperimentConfig(args, feature_dataset_cfg)

    feature_dataset_info = prepare_feature_dataset(exp_cfg)
    tokenized_dataset, feature_datasets = feature_dataset_info

    index_mask = make_index_mask(exp_cfg, tokenized_dataset, feature_datasets)

    start_time = time.perf_counter()
    # note that this fn saves the activations to disk
    get_activation_dataset(
        exp_cfg, model, tokenized_dataset,
        index_mask=index_mask,
        output_precision=args['output_precision']
    )
    total_time = time.perf_counter() - start_time
    print(f'Finished running activations in {total_time / 60 :.2f} minutes')

    # TODO(wesg): if disk flushing rearrage activations to be layer-wise.


def run_neuron_subset_activations(args, model, dataset):
    '''
    Collect activations for a subset of neurons across a dataset

    https://github.com/wesg52/sparse-probing/issues/27
    '''
    if args.neuron_subset is not None:
        neuron_subset = args.neuron_subset
    elif args.neuron_subset_file[-4:] == '.csv':
        neuron_subset = load_neuron_subset_csv(args)
    else:
        raise ValueError(
            f'One of --neuron_subset or --neuron_subset_file must be specified')
    
    if len(neuron_subset) == 0:
        print('No neurons to save activations for. Exiting.')
        return
    start_time = time.perf_counter()
    activation_subset = get_activation_subset(
        model, dataset, neuron_subset,
        batch_size=args.batch_size,
        device=args.device,
        verbose=True
    )
    total_time = time.perf_counter() - start_time
    print(f'Finished running activations in {total_time / 60 :.2f} minutes')

    print('Computing token summary df')
    start_t = time.perf_counter()
    if args.skip_computing_token_summary_df:
        token_summary_df = None
    else:
        token_summary_df = compute_token_summary_dfs(
            activation_subset, dataset, model.tokenizer
        )
    print(
        f'Finished summary df in {(time.perf_counter() - start_t) / 60 :.2f} minutes')

    # save activations metrics and metadata
    experiment_dir = get_experiment_dir(args)
    metadata = get_experiment_metadata(args, total_time)
    metadata['neuron_subset'] = neuron_subset
    save_activation_subset(
        experiment_dir, activation_subset, token_summary_df, metadata,
        output_precision=args.output_precision,
        save_by_neuron=args.save_by_neuron)

    print(f'Saved output files and metadata to {experiment_dir}')


def run_all_activations(args, model, dataset):
    start_time = time.perf_counter()
    activations = get_full_activation_tensor(
        model, dataset,
        batch_size=args.batch_size,
        verbose=True,
        layers=args.layers,
        positions=args.positions,
        flatten_and_ignore_padding=args.flatten_and_ignore_padding
    )
    total_time = time.perf_counter() - start_time

    # save activations metrics and metadata
    experiment_dir = get_experiment_dir(args)
    metadata = get_experiment_metadata(args, total_time)

    save_full_activation_tensor(experiment_dir, activations, metadata,
                                output_precision=args.output_precision)

    print(f'Saved output files and metadata to {experiment_dir}')
    print(f'Finished in {total_time / 60 :.2f} minutes')


def run_activation_histogram(args, model, dataset):
    '''
    Collect activation metrics:
        top k activations for each neuron
        histogram of activations for each neuron

    https://github.com/wesg52/sparse-probing/issues/17
    '''
    start_time = time.perf_counter()
    metrics_params = {k: vars(args)[k] for k in
                      ["top_k", "n_bin", "hist_min", "hist_max"]}
    print(f'\tmetrics params: {metrics_params}')

    top_k_seqix, top_k_pos, bin_counts = get_activation_metrics(
        model, dataset,
        verbose=True,
        batch_size=args.batch_size,
        top_k=args.top_k,
        n_bin=args.n_bin,
        hist_min=args.hist_min,
        hist_max=args.hist_max,
        save_postactivation=args.probe_location.endswith('post'),
    )
    total_time = time.perf_counter() - start_time

    # save activations metrics and metadata
    experiment_dir = get_experiment_dir(args)
    metadata = get_experiment_metadata(args, total_time)
    metadata['metrics_params'] = metrics_params

    save_activation_metrics(
        experiment_dir, top_k_seqix, top_k_pos, bin_counts, metadata,
        output_precision=args.output_precision
    )
    print(f'Saved output files and metadata to {experiment_dir}')
    print(f'Finished in {total_time / 60 :.2f} minutes')


def run_activation_histogram_by_sequence_class():
    # TODO
    pass


if __name__ == '__main__':
    parser = default_argument_parser()
    # base experiment params
    parser.add_argument(
        '--device', choices=['cpu', 'cuda', None],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help='device to use for computation')
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help='batch size to use for model.forward')
    parser.add_argument(
        '--output_precision', type=int, default=8, choices=[8, 16, 32],
        help='number of bits to use for output activations')
    parser.add_argument(
        '--n_threads', type=int,
        default=int(os.getenv('SLURM_CPUS_PER_TASK', 1)),
        help='number of threads to use for pytorch cpu parallelization')
    # params specific to activation histograms
    parser.add_argument(
        '--top_k', type=int, default=25,
        help='number of top activations to collect')
    parser.add_argument(
        '--n_bin', type=int, default=100,
        help='number of histogram bins')
    parser.add_argument(
        '--hist_min', type=float, default=-10,
        help='minimum value for the histogram')
    parser.add_argument(
        '--hist_max', type=float, default=10,
        help='maximum value for the historgram')
    # params specific to activation subset
    parser.add_argument(
        '--neuron_subset', nargs='+', type=parse_neuron_str, default=None,
        help='list of layerix,neuronix pairs to collect activations for')
    parser.add_argument(
        '--neuron_subset_file', default=None,
        help='name of csv file containing a layer,neuron pairs with additional metadata)')
    parser.add_argument(
        '--skip_computing_token_summary_df', action='store_true',
        help='skip computing the token summary df (default compute)')
    parser.add_argument(
        '--save_by_neuron', action='store_true',
        help='save activations by neuron instead of all together in a dict')
    parser.add_argument(
        '--auto_restrict_neuron_subset_file', action='store_true',
        help='automatically restrict the neuron subset file to only include neurons that correspond to the data distribution.')
    # params specific to all activations
    parser.add_argument(
        '--layers', nargs='+', type=int, default=None)
    parser.add_argument(
        '--positions', nargs='+', type=int, default=None)
    parser.add_argument(
        '--flatten_and_ignore_padding', action='store_true')

    args = parser.parse_args()
    print(get_experiment_info_str(args))
    # torch.set_num_threads(args.n_threads)

    model = load_model(args.model, device=args.device)
    dataset = load_feature_dataset(args.feature_dataset)

    print('starting experiment')
    # only allow one experiment type for now
    experiment_type = args.experiment_type[0]
    if experiment_type == 'activation_probe_dataset':
        make_activation_probe_dataset(args, model)
    elif experiment_type == 'all_activations':
        run_all_activations(args, model, dataset)
    elif experiment_type == 'full_activation_histogram':
        run_activation_histogram(args, model, dataset)
    elif experiment_type == 'activation_histogram_by_sequence_class':
        run_activation_histogram_by_sequence_class(args, model, dataset)
    elif experiment_type == 'activation_subset':
        run_neuron_subset_activations(args, model, dataset)
    elif experiment_type == 'test_activation_metrics':
        test_get_activation_metrics(model, dataset)
    elif experiment_type == 'test_activation_subset':
        test_get_activation_subset(model, dataset)
    else:
        raise ValueError(f'Unknown experiment type {experiment_type}')
