import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def token_histogram_by_class(activation_df, logy=False, plot_dist=True, n_bins=50, pos_label='pos', ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    _, bins = np.histogram(activation_df.activation.values, bins=n_bins)
    pos_activations = activation_df.query('label == True').activation.values
    neg_activations = activation_df.query('label == False').activation.values

    if plot_dist:
        count_pos, _ = np.histogram(pos_activations, bins=bins)
        count_neg, _ = np.histogram(neg_activations, bins=bins)
        dist_pos = count_pos / count_pos.sum()
        dist_neg = count_neg / count_neg.sum()
        ax.hist(bins[:-1], bins, weights=dist_pos, alpha=0.5, label=pos_label)
        ax.hist(bins[:-1], bins, weights=dist_neg, alpha=0.5, label='not')
        ax.set_ylabel('empirical distribution')
    else:
        ax.hist(pos_activations, bins=bins, alpha=0.5, label=pos_label)
        ax.hist(neg_activations, bins=bins, alpha=0.5, label='not')
        ax.set_ylabel('count')
    if logy:
        ax.set_yscale('log')
    ax.legend(title='token in', loc='upper right')
    ax.set_xlabel('activation')
    return ax


def token_scatter_by_class(activation_df, pos_label='pos', ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    pos_seq_max = activation_df.query('label == True').groupby(
        'seq_ix').activation.max().values
    pos_seq_mean = activation_df.query('label == True').groupby(
        'seq_ix').activation.mean().values
    neg_seq_max = activation_df.query('label == False').groupby(
        'seq_ix').activation.max().values
    neg_seq_mean = activation_df.query('label == False').groupby(
        'seq_ix').activation.mean().values

    ax.scatter(pos_seq_mean, pos_seq_max, alpha=0.05, label=pos_label, s=10)
    ax.scatter(neg_seq_mean, neg_seq_max, alpha=0.05, label='not', s=10)

    leg = ax.legend(title='sequence', loc='lower right', markerscale=2)
    for lh in leg.legendHandles:
        lh.set_alpha(0.5)
    ax.set_xlabel('mean activation')
    ax.set_ylabel('max activation')
    return ax


def token_disambiguation_boxplot(
        activation_df, decoded_vocab,
        n_top_tokens=20, min_pos_count=50, min_neg_count=50, percentile_cutoff=0.05,
        restrict_alpha=False, min_char_count=0, max_char_count=1000,
        connected=False, ax=None, make_boxes_overlap=False, rotation=0):
    all_activations = activation_df.activation.values
    lb = np.percentile(all_activations, percentile_cutoff)
    ub = np.percentile(all_activations, 100 - percentile_cutoff)

    vocab_df = activation_df.groupby(['token', 'label']).agg(
        {'activation': ['mean', 'count']}
    ).unstack()['activation'].dropna()

    restricted_tokens = set()
    if restrict_alpha:
        for tix, token_string in decoded_vocab.items():
            if not token_string.strip().isalpha():
                restricted_tokens.add(tix)
    if min_char_count > 0 or max_char_count < 1000:
        for tix, token_string in decoded_vocab.items():
            if len(token_string.strip()) < min_char_count or len(token_string) > max_char_count:
                restricted_tokens.add(tix)

    common_vocab_df = vocab_df.loc[
        (vocab_df['count'][True] > min_pos_count) &
        (vocab_df['count'][False] > min_neg_count) &
        (vocab_df.index.map(lambda x: x not in restricted_tokens))
    ]
    token_mean_dif = (
        common_vocab_df['mean'][True] - common_vocab_df['mean'][False]
    )
    most_dif_tokens = token_mean_dif.sort_values(
        ascending=False).head(n_top_tokens).index.values

    common_activation_df = activation_df.loc[
        activation_df['token'].isin(most_dif_tokens)
    ]
    top_pos_medians = common_activation_df.query('label==True').groupby(
        'token').activation.median().sort_values(ascending=False).index.values

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 2))
    if make_boxes_overlap:
        box_width = 0.3
        shift = 0.15
        # Plot the boxes for hue=True
        sns.boxplot(
            x='token', y='activation', data=common_activation_df[common_activation_df['label'] == True],
            showfliers=False, order=top_pos_medians, ax=ax, color='tab:blue',
            whis=(5, 95), width=box_width  # , capprops=whiskerprops_true
        )

        # Shift the x-coordinates of the boxes for hue=True
        for patch in ax.artists[:len(top_pos_medians)]:
            current_x = patch.get_x()
            patch.set_x(current_x - shift)

        # Plot the boxes for hue=False
        sns.boxplot(
            x='token', y='activation', data=common_activation_df[common_activation_df['label'] == False],
            showfliers=False, order=top_pos_medians, ax=ax, color='tab:orange',
            whis=(5, 95), width=box_width  # , capprops=whiskerprops_false
        )

        # Shift the x-coordinates of the boxes for hue=False
        for patch in ax.artists[len(top_pos_medians):]:
            current_x = patch.get_x()
            patch.set_x(current_x + shift)
    else:
        sns.boxplot(
            x='token', y='activation', hue='label', data=common_activation_df,
            hue_order=[True, False], showfliers=False, order=top_pos_medians, ax=ax,
            whis=(5, 95)  # , showmeans=True
        )
    wrap_tokens_with_quotes = True
    ax.set_xticklabels([
        f"'{decoded_vocab[t]}'" if wrap_tokens_with_quotes else decoded_vocab[t]
        for t in top_pos_medians
    ])

    if rotation == 0:
        # Modify the tick label position for every other tick mark on the x-axis
        for index, tick in enumerate(ax.get_xticklabels()):
            tick.set_fontsize(9)
            if index % 2 == 0:
                # Adjust this value to create the desired separation between the tick labels
                tick.set_y(-0.05)
            else:
                tick.set_y(0.01)
        # Modify the tick size for every other tick mark on the x-axis
        for index, tick in enumerate(ax.xaxis.get_major_ticks()):
            if index % 2 == 0:
                # Adjust the length of the tick mark
                tick.tick1line.set_markersize(10)
    else:
        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment('right')
            tick.set_y(0.05)
            tick.set_rotation(30)

    if not connected:
        ax.legend(loc='right', ncols=2)
    else:
        ax.set_ylabel(None)
        ax.spines['left'].set_visible(False)
        # turn off y ticks
        # make y ticks invisible
        # ax.yaxis.set_ticks_position('none')
        # # make y labels invisible
        ax.yaxis.set_ticklabels([])
        # ax.yaxis.set_label_position("right")
        # ax.yaxis.tick_right()
        # turn off legend
        try:
            ax.get_legend().remove()
        except AttributeError:
            pass
    ax.set_ylim([lb, ub])
    ax.grid(axis='y', alpha=0.5, linestyle='--')
    ax.set_xlabel('')
    return ax


def plot_sequence_ablation(loss_df, ax=None, connected=False, rotation=0):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    avg_seq_loss = loss_df.groupby(['label', 'seq_ix']).sum().reset_index()
    avg_seq_loss['loss_increase'] = 100 * \
        (avg_seq_loss['ablated_loss'] - avg_seq_loss['nominal_loss']
         ) / avg_seq_loss['nominal_loss']

    sns.boxplot(
        x='label', y='loss_increase', data=avg_seq_loss, ax=ax,
        whis=(1, 99), showfliers=True, fliersize=1
    )
    if rotation == 0:  # then alternate ticks
        for index, tick in enumerate(ax.get_xticklabels()):
            # set tick font size
            tick.set_fontsize(9)
            if index % 2 == 0:
                # Adjust this value to create the desired separation between the tick labels
                tick.set_y(-0.05)
            else:
                tick.set_y(0.01)
        # Modify the tick size for every other tick mark on the x-axis
        for index, tick in enumerate(ax.xaxis.get_major_ticks()):
            if index % 2 == 0:
                # Adjust the length of the tick mark
                tick.tick1line.set_markersize(10)

    else:
        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment('right')
            tick.set_y(0.05)
            tick.set_rotation(30)

    ub = avg_seq_loss.groupby(
        'label')['loss_increase'].quantile(0.991).max() + 0.01
    lb = avg_seq_loss.groupby(
        'label')['loss_increase'].quantile(0.001).min() - 0.01
    # set just the upper bound
    ax.set_ylim(lb, ub)

    if connected:
        # put axes and label on right side
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")

    ax.set_xlabel(None)
    ax.set_ylabel('sequence loss increase (%)')
    return ax


def token_histogram_by_class_horizontal(activation_df, logy=False, plot_dist=True, n_bins=50, pos_label='pos', percentile_cutoff=0.05, connected=False, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    all_activations = activation_df.activation.values
    lb = np.percentile(all_activations, percentile_cutoff)
    ub = np.percentile(all_activations, 100 - percentile_cutoff)

    _, bins = np.histogram(all_activations, bins=n_bins)
    pos_activations = activation_df.query('label == True').activation.values
    neg_activations = activation_df.query('label == False').activation.values

    if plot_dist:
        count_pos, _ = np.histogram(pos_activations, bins=bins)
        count_neg, _ = np.histogram(neg_activations, bins=bins)
        dist_pos = count_pos / count_pos.sum()
        dist_neg = count_neg / count_neg.sum()
        ax.hist(bins[:-1], bins, weights=dist_pos, alpha=0.5,
                label=pos_label, orientation='horizontal')
        ax.hist(bins[:-1], bins, weights=dist_neg, alpha=0.5,
                label='not', orientation='horizontal')
        ax.set_xlabel('empirical distribution')
    else:
        ax.hist(pos_activations, bins=bins, alpha=0.5,
                label=pos_label, orientation='horizontal')
        ax.hist(neg_activations, bins=bins, alpha=0.5,
                label='not', orientation='horizontal')
        ax.set_xlabel('count')
    if logy:
        ax.set_yscale('log')
    if connected:
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.5, linestyle='--')
    ax.set_ylim(lb, ub)
    ax.legend(title='token', loc='upper right')
    ax.set_ylabel('activation')
    return ax


def plot_context_neuron_row(activation_df, decoded_vocab, pos_label, nix, lix):
    fig, axs = plt.subplots(
        1, 3, figsize=(15, 4),
        gridspec_kw={'width_ratios': [1, 1, 3]}
    )
    token_scatter_by_class(activation_df, ax=axs[0], pos_label=pos_label)
    token_histogram_by_class(activation_df, logy=False,
                             plot_dist=True, n_bins=50, ax=axs[1], pos_label=pos_label)
    axs[1].set_ylim([0, 0.25])
    token_disambiguation_boxplot(
        activation_df, decoded_vocab, n_top_tokens=20, min_pos_count=50, min_neg_count=50, ax=axs[2])
    plt.suptitle(f'{pos_label} neuron: Layer {lix} Neuron {nix}')
    plt.tight_layout()
    plt.show()
