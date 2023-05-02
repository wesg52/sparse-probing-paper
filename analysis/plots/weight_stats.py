import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def weight_boxplot(model_stats, m, ax, tail_cutoff=0.5):
    in_norm = model_stats[m]['in_norm']
    in_bias = model_stats[m]['in_bias']
    biasxnorm = in_norm * in_bias

    inlier_range = np.percentile(biasxnorm, tail_cutoff), np.percentile(
        biasxnorm, 100-tail_cutoff)

    sns.boxplot(data=pd.DataFrame(biasxnorm).T, ax=ax, fliersize=1)
    ax.set_xlabel(f'layer ({m})')
    ax.set_ylabel('$||W_{in}||_2 b_{in}$')
    ax.set_ylim(inlier_range)
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)


def all_model_weight_boxplot(model_stats, figsize=(12, 15)):
    # You can adjust the width and height as needed
    fig = plt.figure(figsize=figsize)
    # Create a GridSpec object with 6 rows and 3 columns
    gs = gridspec.GridSpec(6, 3)

    # Create the first row plots
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1:])

    weight_boxplot(model_stats, 'pythia-70m', ax1)
    weight_boxplot(model_stats, 'pythia-160m', ax2)

    plot_positions = [
        (1, 0, 1), (1, 1, 2),
        (2, 0, 3), (3, 0, 4),
        (4, 0, 5), (5, 0, 6)
    ]

    # fig, axs = plt.subplots(len(models), 1, figsize=(12, 3 * len(models)))
    for m_ix, m in enumerate(model_stats):
        if m_ix <= 1:
            continue
        row, col, plot_num = plot_positions[m_ix - 1]
        ax = plt.subplot(gs[row, :])
        weight_boxplot(model_stats, m, ax)

    plt.suptitle('Distribution of $||W_{in}||_2 b_{in}$ by layer')
    plt.tight_layout()


def plot_normalized_median_norm_bias(models, model_stats, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    for model in models[1:]:
        in_norm = model_stats[model]['in_norm']
        in_bias = model_stats[model]['in_bias']
        n_layers, n_neurons = in_norm.shape

        sp_score = np.median((in_norm * in_bias), axis=1) / \
            np.max(np.median(np.abs(in_norm * in_bias), axis=1))
        relative_depth = np.arange(n_layers) / (n_layers - 1)
        ax.plot(relative_depth, sp_score, label=model.split('-')[-1])

    ax.legend(ncol=2, loc='lower right',
              title='pythia model', fontsize='small')
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('relative layer depth')
    ax.set_ylabel('normalized median($||W_{in}||_2 b_{in}$)')
    ax.set_xlim(-0.005, 1.005)
    # turn off top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
