from load import *
import argparse
import datetime
import os
import torch
import transformer_lens
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial

from activations.activation_subset import load_neuron_subset_csv


@torch.no_grad()
def perform_sequence_ablation(args, model, dataset, neurons, return_logits=False):
    # define hooks
    layer_neurons = {}
    for lix, nix in neurons:
        if lix not in layer_neurons.keys():
            layer_neurons[lix] = []
        layer_neurons[lix].append(nix)

    def neuron_ablation_hook(activations, hook, lix):
        activations[:, :, layer_neurons[lix]] = 0.
        return activations

    hooks = []
    for lix in layer_neurons.keys():
        hooks.append((
            transformer_lens.utils.get_act_name("post", lix),
            partial(neuron_ablation_hook, lix=lix)
        ))

    bs = args.batch_size
    dataloader = DataLoader(
        dataset['tokens'], batch_size=bs, shuffle=False)

    ds_shape = feature_dataset['tokens'].shape
    # n_seqs if save_per_sequence else n_seqs x ctx_len-1
    dataset_losses = torch.zeros(
        ds_shape[0] if args.save_per_sequence
        else (ds_shape[0], ds_shape[1] - 1),
        dtype=torch.float16
    )

    all_logits = torch.empty((len(dataset), model.cfg.d_vocab), dtype=torch.float16)
    for step, batch in enumerate(tqdm(dataloader)):
        batch = batch.to(args.device)

        logits = model.run_with_hooks(
            batch,
            return_type="logits",
            fwd_hooks=hooks,
        )

        if return_logits:
            # for ix in range(batch.shape[0]):
            #     logit_index = dataset['logit_index'][ix].item()
            #     logits16 = logits.detach().cpu().to(torch.float16)[ix, logit_index, :]
            #     all_logits[ix, :] = logits16

            step_start_index = step * bs
            step_end_index = (step + 1) * bs
            logit_indices = dataset['logit_index'][step_start_index:step_end_index]

            logits16 = logits.detach().cpu().to(torch.float16)
            sequences = torch.arange(logits.shape[0])
            all_logits[step_start_index:step_end_index] = logits16[sequences,logit_indices,:]
        token_loss = transformer_lens.utils.lm_cross_entropy_loss(
            logits, batch, per_token=True).detach().cpu().to(torch.float16)

        if args.save_per_sequence:
            dataset_losses[bs*step:bs*(step+1)] = token_loss.mean(axis=1)
        else:
            dataset_losses[bs*step:bs*(step+1), :] = token_loss
    if return_logits:
        return dataset_losses, all_logits
    return dataset_losses


def get_neurons_name(neurons):
    return '_'.join([f'{lix}_{nix}' for lix, nix in neurons])


def run_sequence_ablation_experiment(args, model, dataset):
    if args.neuron_subset is not None:
        neuron_subset = args.neuron_subset
    elif args.neuron_subset_file[-4:] == '.csv':
        # TODO: load from csv
        # neuron_subset = load_neuron_subset_csv(args)
        print('for now must use --neuron_subset')
    else:
        raise ValueError(
            f'One of --neuron_subset or --neuron_subset_file must be specified')

    save_path = os.path.join(
        os.getenv('RESULTS_DIR', 'results'),
        'ablations',
        args.model,
        args.feature_dataset,
    )
    os.makedirs(save_path, exist_ok=True)

    if not args.skip_nominal:
        timestamp = datetime.datetime.now().strftime("%Y:%m:%d:%H:%M:%S")
        print(f'{timestamp } running experiment with no ablations')
        if args.return_logits:
            nominal_loss, nominal_logits = perform_sequence_ablation(
                args, model, dataset, [], return_logits=True)
            torch.save(nominal_logits, os.path.join(save_path, 'nominal_logits.pt'))
        else:
            nominal_loss = perform_sequence_ablation(
                args, model, dataset, [], return_logits=False)
        torch.save(nominal_loss, os.path.join(save_path, 'nominal_loss.pt'))

    for neurons in neuron_subset:
        timestamp = datetime.datetime.now().strftime("%Y:%m:%d:%H:%M:%S")
        # TODO: naming
        print(f'{timestamp } running ablation with neurons: {neurons}')
        if args.return_logits:
            ablated_loss, ablated_logits = perform_sequence_ablation(
                args, model, dataset, neurons, return_logits=True)
            torch.save(ablated_logits, os.path.join(
                save_path, f'ablated_logits_{get_neurons_name(neurons)}.pt'))
        else:
            ablated_loss = perform_sequence_ablation(
                args, model, dataset, neurons, return_logits=False)
        torch.save(ablated_loss, os.path.join(
            save_path, f'ablated_loss_{get_neurons_name(neurons)}.pt'))


def parse_neuron_str(neuron_str: str):
    neurons = []
    for group in neuron_str.split(','):
        lix, nix = group.split('.')
        neurons.append((int(lix), int(nix)))
    return neurons


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--feature_dataset', type=str)
    parser.add_argument('--experiment_type', type=str,
                        default='sequence_ablation')
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help='batch size to use for model.forward')
    parser.add_argument(
        '--device', choices=['cpu', 'cuda', None],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help='device to use for computation')
    parser.add_argument('--save_per_sequence', action='store_true')
    parser.add_argument(
        '--neuron_subset', nargs='+', type=parse_neuron_str, default=None,
        help='list of layerix,neuronix pairs to collect activations for')
    parser.add_argument(
        '--neuron_subset_file', default=None,
        help='name of csv file containing a layer,neuron pairs with additional metadata)')
    parser.add_argument(
        '--auto_restrict_neuron_subset_file', action='store_true',
        help='automatically restrict the neuron subset file to only include neurons that correspond to the data distribution.')
    parser.add_argument(
        '--return_logits', action='store_true',
        help='return logits for each sequence in the dataset')
    parser.add_argument(
        '--skip_nominal', action='store_true',
        help='skip running the nominal experiment')

    args = parser.parse_args()

    model = load_model(args.model, device=args.device)
    feature_dataset = load_feature_dataset(args.feature_dataset)

    if args.experiment_type == 'sequence_ablation':
        run_sequence_ablation_experiment(args, model, feature_dataset)
    else:
        raise ValueError(f'Unknown experiment type {args.experiment_type}')
