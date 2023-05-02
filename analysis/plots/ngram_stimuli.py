import copy
import seaborn as sns
import matplotlib.pyplot as plt


def make_stimuli_plot_data(neuron_stimuli, activation_df, decoded_vocab):
    stimuli_medians = {}
    token_stimulus_ordering = {}
    ngram_activation_dfs = {}

    for t in neuron_stimuli.keys():
        t_adf = copy.deepcopy(activation_df.query('token==@t'))
        t_adf['class_label'] = 'other'  # start with other by default
        stimulus_order = []
        for stimulus in neuron_stimuli[t]:
            n_gram = max(len(p) for p in stimulus)
            for i in range(n_gram, 0, -1):
                t_adf[f'prefix-{i}'] = activation_df.loc[
                    t_adf.index.values - i, 'token'].values

            # use first prefix in the set as class label
            stimulus_string = f"{''.join([decoded_vocab[p] for p in stimulus[0]])}"
            stimulus_order.append(stimulus_string)

            for prefix in stimulus:
                if len(prefix) == 1:
                    t_adf.loc[t_adf['prefix-1'] == prefix[0],
                              'class_label'] = stimulus_string
                elif len(prefix) == 2:
                    t_adf.loc[
                        (t_adf['prefix-2'] == prefix[0]) &
                        (t_adf['prefix-1'] == prefix[1]),
                        'class_label'
                    ] = stimulus_string
                elif len(prefix) == 3:
                    t_adf.loc[
                        (t_adf['prefix-3'] == prefix[0]) &
                        (t_adf['prefix-2'] == prefix[1]) &
                        (t_adf['prefix-1'] == prefix[2]),
                        'class_label'
                    ] = stimulus_string
                else:
                    raise ValueError(
                        f"prefix length {len(prefix)} not supported")

        stimulus_order.append('other')
        token_stimulus_ordering[t] = stimulus_order
        ngram_activation_dfs[t] = t_adf
        # used to order the subplots
        max_class_median_activation = t_adf.groupby(
            'class_label').activation.median().max()
        stimuli_medians[t] = max_class_median_activation

    return ngram_activation_dfs, stimuli_medians, token_stimulus_ordering


def make_neuron_stimulus_plot(ngram_activation_dfs, token_ordering, token_stimulus_ordering, decoded_vocab, title=None):
    fig, axs = plt.subplots(1, len(token_ordering), figsize=(
        1.3 * len(token_ordering), 4.5), sharey=True)
    for ix, t in enumerate(token_ordering):
        t_adf = ngram_activation_dfs[t]
        token_stimuli_order = token_stimulus_ordering[t]
        # see https://stackoverflow.com/questions/46173419/seaborn-change-color-according-to-hue-name
        # always want orange to be the last color
        palette_dict = {
            2: ["C0", "C1"],
            3: ["C0", "C2", "C1"],
            4: ["C0", "C2", "C3", "C1"],
        }
        palette = palette_dict[len(token_stimuli_order)]
        ax = axs[ix]
        sns.boxplot(
            t_adf, x='token', y='activation', hue='class_label',
            hue_order=token_stimuli_order, palette=palette, ax=ax,
            whis=(5, 95), fliersize=1
        )
        ax.legend(loc='lower left', prop={'size': 6}, frameon=False)

        # formatting
        ax.set_xlabel('')
        ax.set_ylabel('pre-activation')
        ax.set_xticklabels([f"'{decoded_vocab[t]}'"])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.grid(axis='y', color='lightgray', linestyle='--', linewidth=0.75)
        ax.tick_params(axis='y', which='both', length=0)
        if ix > 0:
            ax.set_ylabel('')
            ax.spines['left'].set_visible(False)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.suptitle(title, y=0.95, fontsize=16)
    return ax


def make_intro_polysemantic_plot(ngram_activation_dfs, token_ordering, token_stimulus_ordering, decoded_vocab, title=None):
    fig, axs = plt.subplots(1, len(token_ordering), figsize=(
        2 * len(token_ordering), 4), sharey=True)
    for ix, t in enumerate(token_ordering):
        t_adf = ngram_activation_dfs[t]
        token_stimuli_order = token_stimulus_ordering[t]
        # see https://stackoverflow.com/questions/46173419/seaborn-change-color-according-to-hue-name
        # always want orange to be the last color
        palette = ["C0", "C2", "C1"] if len(
            token_stimuli_order) == 3 else ["C0", "C1"]
        ax = axs[ix]
        sns.boxplot(
            t_adf, x='token', y='activation', hue='class_label',
            hue_order=token_stimuli_order, palette=palette, ax=ax,
            whis=(5, 95), fliersize=1
        )
        ax.legend(loc='lower left', prop={'size': 9}, frameon=False)

        # formatting
        ax.set_xlabel('')
        ax.set_ylabel('pre-activation')
        ax.set_xticklabels([f"'{decoded_vocab[t]}'"])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.grid(axis='y', color='lightgray', linestyle='--', linewidth=0.75)
        ax.tick_params(axis='y', which='both', length=0)
        if ix > 0:
            ax.set_ylabel('')
            ax.spines['left'].set_visible(False)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.ylim(-2.5, 3.9)
    plt.suptitle(title, y=0.95)
    return axs
