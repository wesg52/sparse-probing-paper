import os
from tqdm import tqdm
import torch
import einops
import numpy as np
from torch.utils.data import DataLoader
from .quantize import quantize_8bit, unquantize_8bit


# TODO(wesg): add a helper function to delete activations from disk


def get_activation_dataset_path(exp_cfg):
    aggregation = exp_cfg.activation_aggregation \
        if exp_cfg.activation_aggregation is not None \
        else 'none'
    feature_dataset = exp_cfg.feature_dataset \
        if not exp_cfg.probe_next_token_feature \
        else f'{exp_cfg.feature_dataset}.next_token'
    save_path = os.path.join(
        os.environ.get('RESULTS_DIR', 'results'),
        'activation_datasets',
        exp_cfg.model_name,
        feature_dataset,
        f'{exp_cfg.probe_location},{aggregation}'
    )
    return save_path


def load_activation_probing_dataset(exp_cfg, layer, to_numpy=True):
    save_path = get_activation_dataset_path(exp_cfg)
    # TODO(wesg): add support for sharded activations
    file_name = os.path.join(save_path, f'{layer}.all.pt')
    data_tensor = torch.load(file_name)
    if data_tensor.dtype == torch.uint8:
        # unquantize the activations
        quant_info_file_name = os.path.join(
            save_path, f'{layer}.all.quant_info.pt')
        offset, scale = torch.load(quant_info_file_name)
        data_tensor = unquantize_8bit(data_tensor, offset, scale)
    return data_tensor.numpy() if to_numpy else data_tensor


def save_activation_probing_dataset(exp_cfg, layer_activations, output_precision=16, shard=-1):
    save_path = get_activation_dataset_path(exp_cfg)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    shard_name = str(shard).zfill(5) if shard >= 0 else 'all'

    for l, activations in layer_activations.items():

        if output_precision == 8:
            activations, offset, scale = quantize_8bit(activations)
            quant_info = torch.vstack([offset, scale])
            quant_file_name = os.path.join(
                save_path, f'{l}.{shard_name}.quant_info.pt')
            torch.save(quant_info, quant_file_name)

        file_name = os.path.join(save_path, f'{l}.{shard_name}.pt')
        torch.save(activations, file_name)


def make_index_mask(exp_cfg, tokenized_dataset, feature_datasets):
    if exp_cfg.activation_aggregation is None:
        # compute all of the indices which are required for the feature datasets
        all_required_indices = set.union(
            *[set(indices.tolist()) for k, (indices, _) in feature_datasets.items()])
        all_required_indices = np.array(sorted(list(all_required_indices)))
        n, n_ctx = tokenized_dataset['tokens'].shape
        index_mask = np.zeros(n * n_ctx).astype(bool)
        index_mask[all_required_indices] = True

    else:
        # get valid indices to average/max over
        index_mask = sum([feature_datasets[name][0]
                         for name in feature_datasets])
        index_mask = index_mask > 0
    return index_mask


def process_activation_batch(exp_cfg, batch_activations, step, index_mask=None):
    cur_batch_size = batch_activations.shape[0]

    if exp_cfg.activation_aggregation is None:
        # only save the activations for the required indices
        offset = step * exp_cfg.batch_size * batch_activations.shape[1]
        batch_end = cur_batch_size * batch_activations.shape[1]
        batch_activations = einops.rearrange(
            batch_activations, 'b c d -> (b c) d')  # batch, context, dim
        processed_activations = batch_activations[index_mask[offset:offset+batch_end]]

    elif exp_cfg.activation_aggregation == 'mean':
        # average over the context dimension for valid tokens only
        offset = step * exp_cfg.batch_size
        batch_mask = index_mask[offset: offset+cur_batch_size, :, None]
        masked_activations = batch_activations * batch_mask
        batch_valid_ixs = index_mask[offset:offset+cur_batch_size].sum(dim=1)
        processed_activations = masked_activations.sum(
            dim=1) / batch_valid_ixs[:, None]

    elif exp_cfg.activation_aggregation == 'max':
        # max over the context dimension for valid tokens only (set invalid tokens to -1)
        offset = step * exp_cfg.batch_size
        batch_mask = index_mask[offset: offset+cur_batch_size, :, None].to(int)
        # set masked tokens to -1
        masked_activations = batch_activations * batch_mask + (batch_mask - 1)
        processed_activations = masked_activations.max(dim=1)[0]

    return processed_activations


def save_activation(tensor, hook):
    hook.ctx['activation'] = tensor.detach().cpu().to(torch.float16)


@torch.no_grad()
def get_activation_dataset(
    exp_cfg, model, tokenized_dataset,
    layers='all', disk_flush_size=-1, index_mask=None, device=None, output_precision=16
):
    if layers == 'all':
        layers = list(range(model.cfg.n_layers))
    if device is None:
        device = model.cfg.device

    hooks = [
        (f'blocks.{layer_ix}.{exp_cfg.probe_location}', save_activation)
        for layer_ix in layers
    ]

    n_seq, ctx_len = tokenized_dataset['tokens'].shape
    activation_rows = sum(index_mask) \
        if exp_cfg.activation_aggregation is None \
        else n_seq
    activation_rows = min(activation_rows, disk_flush_size) \
        if disk_flush_size > 0 \
        else activation_rows

    layer_activations = {
        l: torch.zeros(activation_rows, model.cfg.d_mlp, dtype=torch.float16)
        for l in layers
    }
    layer_offsets = {l: 0 for l in layers}
    bs = exp_cfg.batch_size
    dataloader = DataLoader(
        tokenized_dataset['tokens'], batch_size=bs, shuffle=False)

    for step, batch in enumerate(tqdm(dataloader, disable=False)):
        model.run_with_hooks(
            batch.to(device),
            fwd_hooks=hooks,
            stop_at_layer=max(layers) + 1,
        )

        for lix, (hook_pt, _) in enumerate(hooks):
            batch_activations = model.hook_dict[hook_pt].ctx['activation']

            processed_activations = process_activation_batch(
                exp_cfg, batch_activations, step, index_mask)

            offset = layer_offsets[layers[lix]]
            save_rows = processed_activations.shape[0]

            layer_activations[layers[lix]][
                offset:offset+save_rows] = processed_activations

            layer_offsets[layers[lix]] += save_rows

        model.reset_hooks()

        # TODO(wesg): intermediate disk flushing

    save_activation_probing_dataset(
        exp_cfg, layer_activations, output_precision)
