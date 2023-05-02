import matplotlib as mpl
import math
import matplotlib.pyplot as plt
import numpy as np


# fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
def colorFader(c1='red', c2='blue', mix=0):
    c1 = np.array(mpl.colors.to_rgb(c1))
    c2 = np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)


def plot_metric_over_sparsity_per_layer(rdf, features=(), metric='test_pr_auc'):
    if len(features) == 0:
        features = sorted(rdf.index.get_level_values(0).unique())

    layers = sorted(rdf.index.get_level_values(1).unique())
    ks = sorted(rdf.index.get_level_values(2).unique())
    n_rows = math.ceil(len(layers)/4)
    fig, axs = plt.subplots(n_rows, 4, figsize=(10, 2.5*n_rows), sharey=True)
    for l in layers:
        ax = axs[l//4, l % 4]
        for f in features:
            perf = rdf.loc[f, l, :][metric]
            ax.plot(ks, perf, label=f)
            ax.scatter(ks, perf, s=2)

        ax.set_xscale('log')
        ax.set_title(f'layer {l}')
        if l % 4 == 0:
            ax.set_ylabel(metric)
        if l//4 == n_rows-1:
            ax.set_xlabel('sparsity')
        if l == 3:
            ax.legend()
    plt.tight_layout()


def plot_layer_metric_over_sparsity_per_feature(rdf, features=(), metric='test_pr_auc', n_cols=3):
    if len(features) == 0:
        features = sorted(rdf.index.get_level_values(0).unique())

    layers = sorted(rdf.index.get_level_values(1).unique())
    ks = sorted(rdf.index.get_level_values(2).unique())
    n_rows = math.ceil(len(features) / n_cols)
    fig, axs = plt.subplots(
        n_rows, n_cols, figsize=(10, 2.5*n_rows+5), sharey=True)
    for f_ix, f in enumerate(features):
        ax = axs[f_ix//n_cols, f_ix % n_cols] if n_rows > 1 else axs[f_ix % n_cols]
        for l in layers:
            perf = rdf.loc[f, l, :][metric]
            ax.plot(ks, perf, label=l, color=colorFader(mix=l/len(layers)))
            ax.scatter(ks, perf, s=2, color=colorFader(mix=l/len(layers)))

        ax.set_xscale('log')
        ax.set_title(f'feature {f}')
        if f_ix % n_cols == 0:
            ax.set_ylabel(metric)
        if f_ix//n_cols == 2:
            ax.set_xlabel('sparsity')
        if f_ix % n_cols == 0 and f_ix//n_cols == n_rows-1:
            ax.legend(ncols=4, bbox_to_anchor=(0, -1, 1, 1), loc='lower left')
    plt.tight_layout()
