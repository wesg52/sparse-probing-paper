import argparse
import os
import torch
import numpy as np
from load import load_model


def compute_and_save_weight_statistics(args):
    model = load_model(args.model_name, device='cpu')

    in_norm = model.W_in.norm(dim=1).numpy()
    in_bias = model.b_in.numpy()
    out_norm = model.W_out.norm(dim=-1).numpy()
    out_bias = model.b_out.numpy()
    cos = torch.nn.CosineSimilarity()(model.W_in, torch.swapaxes(model.W_out, 1, 2))

    n_layers, n_neurons = in_norm.shape
    statistics = np.zeros((5, n_layers, n_neurons))
    statistics[0] = in_norm
    statistics[1] = in_bias
    statistics[2] = out_norm
    statistics[3, :, :len(out_bias[0])] = out_bias
    statistics[4] = cos

    save_dir = os.path.join(
        os.environ.get('RESULTS_DIR', 'results'),
        'weight_statistics'
    )
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, f'{args.model_name}.npy')
    np.save(save_file, statistics)


def load_weight_statistics(model_name, save_dir=None):
    if save_dir is None:
        save_dir = os.path.join(
            os.environ.get('RESULTS_DIR', 'results'),
            'weight_statistics'
        )
    stats = np.load(os.path.join(save_dir, f'{model_name}.npy'))
    _, _, n_neurons = stats.shape
    return {
        'in_norm': stats[0],
        'in_bias': stats[1],
        'out_norm': stats[2],
        'out_bias': stats[3, :, :n_neurons//4],
        'cos': stats[4]
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='Model name')
    args = parser.parse_args()
    compute_and_save_weight_statistics(args)
