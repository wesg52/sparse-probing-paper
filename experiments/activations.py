import torch
import einops
from torch.utils.data import DataLoader
from datasets import Dataset


def save_activation(tensor, hook):
    hook.ctx['activation'] = tensor.detach().cpu().to(torch.float16)


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
        masked_activations = batch_activations * batch_mask + (batch_mask - 1)
        processed_activations = batch_activations.max(dim=1)[0]

    return processed_activations


@torch.no_grad()
def get_activation_dataset(
    exp_cfg, model, text_dataset, layer_ix, index_mask=None
):
    hook_pt = f'blocks.{layer_ix}.{exp_cfg.probe_location}'

    dataloader = DataLoader(
        text_dataset['tokens'], batch_size=exp_cfg.batch_size, shuffle=False)
    layer_activations = []

    for step, batch in enumerate(dataloader):
        model.run_with_hooks(
            batch,
            fwd_hooks=[(hook_pt, save_activation)],
            stop_at_layer=layer_ix + 1,
        )

        batch_activations = model.hook_dict[hook_pt].ctx['activation']

        processed_activations = process_activation_batch(
            exp_cfg, batch_activations, step, index_mask)

        layer_activations.append(processed_activations)
        model.reset_hooks()

    activation_dataset = torch.concat(layer_activations, dim=0).numpy()
    return activation_dataset
