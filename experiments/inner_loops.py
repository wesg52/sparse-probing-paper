import time
import math
import copy
import numpy as np

from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from .metrics import get_binary_cls_perf_metrics
from .probes import *
from .regression_inner_loops import dense_regression_probe, heuristic_sparse_regression_sweep


def make_k_list(d_max, max_k=None):
    if max_k is None:
        max_k = d_max
    base_ks = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12]
    exp_ks = list(16 * 2**np.arange(int(math.log2(d_max))-3))
    if exp_ks[-1] != d_max:
        exp_ks.append(d_max)
    return [k for k in base_ks + exp_ks if k <= max_k]


def split_and_preprocess(exp_cfg, activation_dataset, feature_labels):
    X_train, X_test, y_train, y_test = train_test_split(
        activation_dataset, feature_labels,
        test_size=exp_cfg.test_set_frac, random_state=exp_cfg.seed)

    if exp_cfg.normalize_activations:
        neuron_std = np.maximum(X_train.std(axis=0), 1e-3)
        # x10 for numerical stability (most neurons have std < 1)
        X_train /= (neuron_std * 10)
        X_test /= (neuron_std * 10)

    return X_train, X_test, y_train, y_test


def dense_probe(exp_cfg, activation_dataset, feature_labels):
    """
    Train a dense probe on the activation dataset.

    Parameters
    ----------
    exp_cfg : as specified by the CLI in probing_experiment.py
    activation_dataset : np.ndarray (n_samples, n_neurons)
    feature_labels : np.ndarray (n_samples) with labels -1 or +1.
    """
    X_train, X_test, y_train, y_test = split_and_preprocess(
        exp_cfg, activation_dataset, feature_labels)

    lr = LogisticRegression(
        class_weight='balanced',
        penalty='l2',
        solver='saga',
        n_jobs=-1,
        max_iter=200
    )
    start_t = time.time()
    lr = lr.fit(X_train, y_train)
    elapsed_time = start_t - time.time()
    lr_score = lr.decision_function(X_test)
    lr_pred = lr.predict(X_test)

    results = get_binary_cls_perf_metrics(y_test, lr_pred, lr_score)
    results['elapsed_time'] = elapsed_time
    results['coef'] = lr.coef_[0]
    return results


def enumerate_monosemantic(exp_cfg, activation_dataset, feature_labels, top_k=30):
    """
    Test the classification performance of the top 30 neurons as ordered by
    their f_statistic (ratio of class variances).

    Parameters
    ----------
    exp_cfg : as specified by the CLI in probing_experiment.py
    activation_dataset : np.ndarray (n_samples, n_neurons)
    feature_labels : np.ndarray (n_samples) with labels -1 or +1.
    """
    X_train, X_test, y_train, y_test = split_and_preprocess(
        exp_cfg, activation_dataset, feature_labels)

    neuron_ranking = get_heuristic_neuron_ranking(
        X_train, y_train, exp_cfg.heuristic_feature_selection_method)
    top_neurons = sorted(neuron_ranking[-top_k:])

    neuron_results = {}
    for n in top_neurons:
        support_indices = [n]
        n_lr = LogisticRegression(
            class_weight='balanced',
            penalty='l2',
            solver='saga',
            n_jobs=-1,
            max_iter=200
        )
        n_lr.fit(X_train[:, support_indices], y_train)
        n_lr_score = n_lr.decision_function(X_test[:, support_indices])
        n_lr_pred = n_lr.predict(X_test[:, support_indices])

        results = get_binary_cls_perf_metrics(y_test, n_lr_pred, n_lr_score)
        results['elapsed_time'] = 0
        results['coef'] = n_lr.coef_[0]
        neuron_results[n] = results

    return neuron_results


def osp_tuning(exp_cfg, activation_dataset, feature_labels):
    """
    Hyperparameter tuning for the optimal sparse probe.

    Parameters
    ----------
    exp_cfg : as specified by the CLI in probing_experiment.py
    activation_dataset : np.ndarray (n_samples, n_neurons)
    feature_labels : np.ndarray (n_samples) with labels -1 or +1.
    """
    X_train, _, _, _ = train_test_split(
        activation_dataset, feature_labels,
        test_size=0.8, random_state=exp_cfg.seed)

    n, _ = X_train.shape
    osp_heuristic_filter_sizes = [50]
    regs = 1/(n * (1-exp_cfg.test_set_frac))**np.linspace(0.75, 1/3, 6)
    parameter_results = {}
    for r in regs:
        for f in osp_heuristic_filter_sizes:
            exp_cfg.osp_heuristic_filter_size = f
            exp_cfg.seed += 1
            parameter_results[(r, f)] = optimal_sparse_probing(
                exp_cfg, activation_dataset, feature_labels, regularization=r)

    return parameter_results


def optimal_sparse_probing(exp_cfg, activation_dataset, feature_labels, regularization=None):
    """
    Train an optimal sparse probe for various values of k (use telescoping for warm starts).

    Parameters
    ----------
    exp_cfg : as specified by the CLI in probing_experiment.py
    activation_dataset : np.ndarray (n_samples, n_neurons)
    feature_labels : np.ndarray (n_samples) with labels -1 or +1.
    regularization : float, optional defaults to 1/n**0.5
    """
    X_train, X_test, y_train, y_test = split_and_preprocess(
        exp_cfg, activation_dataset, feature_labels)

    n, d_act = X_train.shape

    if regularization is None:
        regularization = 1/n**0.5

    # score is equal to the mean difference
    pos_class = y_train == 1
    scores = np.abs(X_train[pos_class].mean(axis=0) -
                    X_train[~pos_class].mean(axis=0))
    coef_filter = np.sort(
        scores.argsort()[-exp_cfg.osp_heuristic_filter_size:])
    scores = scores[coef_filter]

    class_weights = get_balanced_class_weights(y_train)

    ks = make_k_list(d_act, exp_cfg.osp_upto_k)

    layer_results = {}
    for k in ks[::-1]:  # iterate in descending order
        # warm start - set max_k highest scores to 1
        s0 = np.zeros_like(coef_filter)
        s0[np.argsort(scores)[-k:]] = 1

        model_stats, filtered_support, beta, bias = sparse_classification_oa(
            X_train[:, coef_filter], y_train, k,
            regularization, s0=s0, weights=class_weights,
            time_limit=exp_cfg.gurobi_timeout, verbose=exp_cfg.gurobi_verbose
        )
        support = coef_filter[filtered_support]
        y_score = X_test[:, support] @ beta + bias
        y_pred = np.sign(y_score)
        osp_perf = get_binary_cls_perf_metrics(y_test, y_pred, y_score)
        layer_results[('OSP', k)] = {**osp_perf, **model_stats}
        layer_results[('OSP', k)]['support'] = support
        layer_results[('OSP', k)]['beta'] = beta
        layer_results[('OSP', k)]['bias'] = bias

        # Use recovered features to train a logistic regression model
        k_lr = LogisticRegression(
            class_weight='balanced',
            penalty='l2',
            solver='saga',
            n_jobs=-1,
            max_iter=200
        )
        start_t = time.time()
        k_lr = k_lr.fit(X_train[:, support], y_train)
        k_lr_elapsed_time = time.time() - start_t
        k_lr_score = k_lr.decision_function(X_test[:, support])
        k_lr_pred = k_lr.predict(X_test[:, support])
        layer_results[('LR', k)] = get_binary_cls_perf_metrics(
            y_test, k_lr_pred, k_lr_score)
        layer_results[('LR', k)]['runtime'] = k_lr_elapsed_time
        layer_results[('LR', k)]['support'] = support
        layer_results[('LR', k)]['beta'] = k_lr.coef_[0]
        layer_results[('LR', k)]['bias'] = k_lr.intercept_[0]

        scores = np.zeros_like(scores)
        scores[filtered_support] = np.abs(k_lr.coef_[0])

    return layer_results


def osp_iterative_pruning(exp_cfg, activation_dataset, feature_labels, regularization=None):
    """
    Iteratively prune features from optimal sparse probe to test ensemble vs. superposition.

    Parameters
    ----------
    exp_cfg : as specified by the CLI in probing_experiment.py
    activation_dataset : np.ndarray (n_samples, n_neurons)
    feature_labels : np.ndarray (n_samples) with labels -1 or +1.
    regularization : float, optional defaults to 10/n
    """
    X_train, X_test, y_train, y_test = split_and_preprocess(
        exp_cfg, activation_dataset, feature_labels)

    n, d_act = X_train.shape

    if regularization is None:
        regularization = 10/n

    # score is equal to the f_statistic (higher is better)
    scores, _ = f_classif(X_train, y_train)
    scores_copy = copy.deepcopy(scores)

    static_k = exp_cfg.iterative_pruning_fixed_k
    n_prune_steps = exp_cfg.iterative_pruning_n_prune_steps

    layer_results = {'pruned_neurons': [], 'pruned_neurons_f1': []}
    mono_neurons_scores = {}
    for i in range(n_prune_steps):
        class_weights = get_balanced_class_weights(y_train)

        # filter features to only be top features according to f_statistic
        coef_filter = np.sort(
            scores.argsort()[-exp_cfg.osp_heuristic_filter_size:])

        s0 = np.zeros(exp_cfg.osp_heuristic_filter_size)
        model_stats, filtered_support, beta, bias = sparse_classification_oa(
            X_train[:, coef_filter], y_train, static_k, regularization, s0=s0, weights=class_weights,
            time_limit=exp_cfg.gurobi_timeout, verbose=exp_cfg.gurobi_verbose
        )
        support = coef_filter[filtered_support]
        y_score = X_test[:, support] @ beta + bias
        y_pred = np.sign(y_score)
        osp_perf = get_binary_cls_perf_metrics(y_test, y_pred, y_score)
        layer_results[('OSP', i)] = {**osp_perf, **model_stats}
        layer_results[('OSP', i)]['support'] = support
        layer_results[('OSP', i)]['beta'] = beta
        layer_results[('OSP', i)]['bias'] = bias

        # Use recovered features to train a logistic regression model
        k_lr = LogisticRegression(
            class_weight='balanced',
            penalty='l2',
            solver='saga',
            n_jobs=-1,
            max_iter=200
        )
        start_t = time.time()
        k_lr = k_lr.fit(X_train[:, support], y_train)
        k_lr_elapsed_time = time.time() - start_t
        k_lr_score = k_lr.decision_function(X_test[:, support])
        k_lr_pred = k_lr.predict(X_test[:, support])
        layer_results[('LR', i)] = get_binary_cls_perf_metrics(
            y_test, k_lr_pred, k_lr_score)
        layer_results[('LR', i)]['runtime'] = k_lr_elapsed_time
        layer_results[('LR', i)]['support'] = support
        layer_results[('LR', i)]['beta'] = k_lr.coef_[0]
        layer_results[('LR', i)]['bias'] = k_lr.intercept_[0]

        # Train a logistic regression model on each of the support individually
        support_scores = np.zeros(len(support))
        for j, s in enumerate(support):
            if s in mono_neurons_scores:
                support_scores[j] = mono_neurons_scores[s]
                continue  # already computed since single neuron score is unchanged
            lr = LogisticRegression(
                class_weight='balanced',
                penalty='l2',
                solver='saga',
                n_jobs=-1,
                max_iter=200
            )
            start_t = time.time()
            lr = lr.fit(X_train[:, s].reshape(-1, 1), y_train)
            lr_elapsed_time = time.time() - start_t
            lr_score = lr.decision_function(X_test[:, s].reshape(-1, 1))
            lr_pred = lr.predict(X_test[:, s].reshape(-1, 1))
            lr_perf = get_binary_cls_perf_metrics(y_test, lr_pred, lr_score)
            layer_results[('mono', s)] = {
                **lr_perf, 'runtime': lr_elapsed_time}
            layer_results[('mono', s)]['support'] = s
            layer_results[('mono', s)]['beta'] = lr.coef_[0]
            layer_results[('mono', s)]['bias'] = lr.intercept_[0]
            mono_neurons_scores[s] = lr_perf['test_f1_score']
            support_scores[j] = lr_perf['test_f1_score']
        best_neuron = support[np.argmax(support_scores)]
        scores[best_neuron] = 0  # remove neuron from future consideration
        layer_results['pruned_neurons'].append(best_neuron)
        layer_results['pruned_neurons_f1'].append(np.max(support_scores))

    layer_results['f_scores'] = scores_copy
    return layer_results


def heuristic_sparsity_sweep(exp_cfg, activation_dataset, feature_labels):
    """
    Train a heuristic sparse probe for various values of k.

    Parameters
    ----------
    exp_cfg : as specified by the CLI in probing_experiment.py
    activation_dataset : np.ndarray (n_samples, n_neurons)
    feature_labels : np.ndarray (n_samples) with labels -1 or +1.
    """
    X_train, X_test, y_train, y_test = split_and_preprocess(
        exp_cfg, activation_dataset, feature_labels)

    n, d_act = X_train.shape
    neuron_ranking = get_heuristic_neuron_ranking(
        X_train, y_train, exp_cfg.heuristic_feature_selection_method)

    ks = make_k_list(d_act, exp_cfg.max_k)

    layer_results = {}
    for k in ks:
        support_indices = np.sort(neuron_ranking[-k:])

        k_lr = LogisticRegression(
            class_weight='balanced',
            penalty='l2',
            solver='saga',
            n_jobs=-1,
            max_iter=200
        )
        start_t = time.time()
        k_lr = k_lr.fit(X_train[:, support_indices], y_train)
        k_lr_elapsed_time = time.time() - start_t

        k_lr_score = k_lr.decision_function(X_test[:, support_indices])
        k_lr_pred = k_lr.predict(X_test[:, support_indices])

        layer_results[k] = get_binary_cls_perf_metrics(
            y_test, k_lr_pred, k_lr_score)
        layer_results[k]['elapsed_time'] = k_lr_elapsed_time
        layer_results[k]['support'] = support_indices
        layer_results[k]['coef'] = k_lr.coef_[0]

    return layer_results


def fast_heuristic_sparsity_sweep(exp_cfg, activation_dataset, feature_labels):
    """
    Train a heuristic sparse probe based on f_statistic for smaller values of k.

    Parameters
    ----------
    exp_cfg : as specified by the CLI in probing_experiment.py
    activation_dataset : np.ndarray (n_samples, n_neurons)
    feature_labels : np.ndarray (n_samples) with labels -1 or +1.
    """
    X_train, X_test, y_train, y_test = split_and_preprocess(
        exp_cfg, activation_dataset, feature_labels)

    n, d_act = X_train.shape

    neuron_ranking = get_heuristic_neuron_ranking(
        X_train, y_train, exp_cfg.heuristic_feature_selection_method)

    ks = make_k_list(d_act, exp_cfg.max_k)

    layer_results = {}
    for k in ks:
        support_indices = np.sort(neuron_ranking[-k:])

        k_lr = LogisticRegression(
            class_weight='balanced',
            penalty='l2',
            solver='saga',
            n_jobs=-1,
            max_iter=250
        )
        start_t = time.time()
        k_lr = k_lr.fit(X_train[:, support_indices], y_train)
        k_lr_elapsed_time = time.time() - start_t

        k_lr_score = k_lr.decision_function(X_test[:, support_indices])
        k_lr_pred = k_lr.predict(X_test[:, support_indices])

        layer_results[k] = get_binary_cls_perf_metrics(
            y_test, k_lr_pred, k_lr_score)
        layer_results[k]['elapsed_time'] = k_lr_elapsed_time
        layer_results[k]['support'] = support_indices
        layer_results[k]['coef'] = k_lr.coef_[0]

    return layer_results


def telescopic_sparsity_sweep(exp_cfg, activation_dataset, feature_labels, reg=-1, l1_ratio=-1):
    """
    Train a telescopic probe for various values of k.

    That is, use a cheap heuristic to select the max_k neurons, then use
    a more expensive heuristic to select the best k of those in descending order.

    Parameters
    ----------
    exp_cfg : as specified by the CLI in probing_experiment.py
    activation_dataset : np.ndarray (n_samples, n_neurons)
    feature_labels : np.ndarray (n_samples) with labels -1 or +1.
    """
    X_train, X_test, y_train, y_test = split_and_preprocess(
        exp_cfg, activation_dataset, feature_labels)

    n, d_act = X_train.shape

    neuron_ranking = get_heuristic_neuron_ranking(
        X_train, y_train, exp_cfg.heuristic_feature_selection_method)

    ks = make_k_list(d_act, exp_cfg.max_k)

    layer_results = {}
    for k in ks[::-1]:
        support_indices = np.sort(neuron_ranking[-k:])
        k_lr = LogisticRegression(
            class_weight='balanced',
            penalty='elasticnet',
            solver='saga',
            C=reg if reg > 0 else k / 20,
            l1_ratio=l1_ratio if l1_ratio > 0 else np.clip(
                2.718**(-k/20), 0.1, 0.9),
            n_jobs=-1,
            max_iter=1000,
            warm_start=True
        )
        start_t = time.time()
        k_lr = k_lr.fit(X_train[:, support_indices], y_train)
        k_lr_elapsed_time = time.time() - start_t

        k_lr_score = k_lr.decision_function(X_test[:, support_indices])
        k_lr_pred = k_lr.predict(X_test[:, support_indices])
        k_lr_prob = k_lr.predict_proba(X_test[:, support_indices])

        layer_results[k] = get_binary_cls_perf_metrics(
            y_test, k_lr_pred, k_lr_score)
        layer_results[k]['test_log_loss'] = log_loss(y_test, k_lr_prob)
        layer_results[k]['elapsed_time'] = k_lr_elapsed_time
        layer_results[k]['support'] = support_indices
        layer_results[k]['coef'] = k_lr.coef_[0]
        layer_results[k]['bias'] = k_lr.intercept_[0]
        layer_results[k]['max_iter'] = k_lr.n_iter_[0]

        scores = np.zeros(d_act)
        scores[support_indices] = np.abs(k_lr.coef_[0])
        neuron_ranking = np.argsort(scores)

    return layer_results


def tune_telescoping_sparsity_sweep(exp_cfg, activation_dataset, feature_labels):
    """
    Tune the regularization parameter for a telescopic probe.

    Parameters
    ----------
    exp_cfg : as specified by the CLI in probing_experiment.py
    activation_dataset : np.ndarray (n_samples, n_neurons)
    feature_labels : np.ndarray (n_samples) with labels -1 or +1.
    """
    ratios = np.linspace(0.1, 0.9, 5)
    regs = np.logspace(-2, 2, 5)
    tune_results = {}
    for l1_ratio in ratios:
        for reg in regs:
            layer_results = telescopic_sparsity_sweep(
                exp_cfg, activation_dataset, feature_labels, reg, l1_ratio)
            tune_results[(l1_ratio, reg)] = layer_results
    return tune_results


def rotation_baseline(exp_cfg, activation_dataset, feature_labels):
    """
    Train a logistic regression model on the rotated activations.

    Parameters
    ----------
    exp_cfg : as specified by the CLI in probing_experiment.py
    activation_dataset : np.ndarray (n_samples, n_neurons)
    feature_labels : np.ndarray (n_samples) with labels -1 or +1.
    """

    n, d_act = activation_dataset.shape

    ks = make_k_list(d_act, exp_cfg.max_k)

    layer_results = {}
    for k in ks:
        n_samples = round(16 / k**0.5)
        for i in range(n_samples):
            A = np.random.normal(size=(d_act, k))
            U, _, _ = np.linalg.svd(A, full_matrices=False)

            X_train, X_test, y_train, y_test = split_and_preprocess(
                exp_cfg, activation_dataset @ U, feature_labels)

            k_lr = LogisticRegression(
                class_weight='balanced',
                penalty='l2',
                solver='saga',
                n_jobs=-1,
                max_iter=500
            )
            start_t = time.time()
            k_lr = k_lr.fit(X_train, y_train)
            k_lr_elapsed_time = time.time() - start_t

            k_lr_score = k_lr.decision_function(X_test)
            k_lr_pred = k_lr.predict(X_test)

            layer_results[(k, i)] = get_binary_cls_perf_metrics(
                y_test, k_lr_pred, k_lr_score)
            layer_results[(k, i)]['elapsed_time'] = k_lr_elapsed_time
    return layer_results


def rotation_baseline_dxd(exp_cfg, activation_dataset, feature_labels, n_trials=5):
    """
    Train a logistic regression model on the rotated activations.

    Parameters
    ----------
    exp_cfg : as specified by the CLI in probing_experiment.py
    activation_dataset : np.ndarray (n_samples, n_neurons)
    feature_labels : np.ndarray (n_samples) with labels -1 or +1.
    """
    _, d = activation_dataset.shape
    results = {}
    for i in range(n_trials):
        A = np.random.normal(size=(d, d))

        results[i] = telescopic_sparsity_sweep(
            exp_cfg, activation_dataset @ A, feature_labels)

    return results


def iterative_pruning_rotation_baseline_dxd(exp_cfg, activation_dataset, feature_labels, top_k=100, n_trials=5):
    """
    Train a logistic regression model on the rotated activations.

    Parameters
    ----------
    exp_cfg : as specified by the CLI in probing_experiment.py
    activation_dataset : np.ndarray (n_samples, n_neurons)
    feature_labels : np.ndarray (n_samples) with labels -1 or +1.
    """
    X_train, X_test, y_train, y_test = split_and_preprocess(
        exp_cfg, activation_dataset, feature_labels)

    _, d = activation_dataset.shape
    baseline_results = {}
    for i in range(n_trials):

        A = np.random.normal(size=(d, d))
        X_train_rot = X_train @ A
        X_test_rot = X_test @ A

        neuron_ranking = get_heuristic_neuron_ranking(
            X_train_rot, y_train, exp_cfg.heuristic_feature_selection_method)
        top_neurons = sorted(neuron_ranking[-top_k:])

        neuron_results = {}
        for n in top_neurons:
            support_indices = [n]
            n_lr = LogisticRegression(
                class_weight='balanced',
                penalty='l2',
                solver='saga',
                n_jobs=-1,
                max_iter=100
            )
            n_lr.fit(X_train_rot[:, support_indices], y_train)
            n_lr_score = n_lr.decision_function(X_test_rot[:, support_indices])
            n_lr_pred = n_lr.predict(X_test_rot[:, support_indices])

            results = get_binary_cls_perf_metrics(
                y_test, n_lr_pred, n_lr_score)
            results['elapsed_time'] = 0
            results['coef'] = n_lr.coef_[0]
            neuron_results[n] = results
        baseline_results[i] = neuron_results

    # nominal results
    neuron_ranking = get_heuristic_neuron_ranking(
        X_train, y_train, exp_cfg.heuristic_feature_selection_method)
    top_neurons = sorted(neuron_ranking[-top_k:])

    nominal_results = {}
    for n in top_neurons:
        support_indices = [n]
        n_lr = LogisticRegression(
            class_weight='balanced',
            penalty='l2',
            solver='saga',
            n_jobs=-1,
            max_iter=200
        )
        n_lr.fit(X_train[:, support_indices], y_train)
        n_lr_score = n_lr.decision_function(X_test[:, support_indices])
        n_lr_pred = n_lr.predict(X_test[:, support_indices])

        results = get_binary_cls_perf_metrics(y_test, n_lr_pred, n_lr_score)
        results['elapsed_time'] = 0
        results['coef'] = n_lr.coef_[0]
        nominal_results[n] = results

    return {'baseline': baseline_results, 'nominal': nominal_results}


def test_osp_heuristic_filtering(exp_cfg, activation_dataset, feature_labels):
    """
    Test performance of OSP with different heuristic filtering strategies.

    Parameters
    ----------
    exp_cfg : as specified by the CLI in probing_experiment.py
    activation_dataset : np.ndarray (n_samples, n_neurons)
    feature_labels : np.ndarray (n_samples) with labels -1 or +1.
    """
    X_train, X_test, y_train, y_test = split_and_preprocess(
        exp_cfg, activation_dataset, feature_labels)

    n, d_act = X_train.shape

    lr = LogisticRegression(
        class_weight='balanced',
        penalty='elasticnet',
        l1_ratio=0.8,
        solver='saga',
        n_jobs=-1)
    lr = lr.fit(X_train, y_train)
    top_coefs = np.argsort(np.abs(lr.coef_[0]))
    coef_filter = np.sort(top_coefs[-exp_cfg.osp_heuristic_filter_size:])

    mi = mutual_info_classif(X_train, y_train)

    f_stat, p_val = f_classif(X_train, y_train)

    pos_class = y_train == 1
    mean_dif = X_train[pos_class, :].mean(
        axis=0) - X_train[~pos_class, :].mean(axis=0)

    ks = make_k_list(d_act, exp_cfg.max_k)

    osp_results = {}
    for k in ks:
        if k > exp_cfg.osp_upto_k:
            continue

        class_weights = get_balanced_class_weights(y_train)

        # Unfiltered
        s0 = np.zeros(d_act+1)
        s0[top_coefs[-k:]] = 1
        model_stats, support, beta, bias = sparse_classification_oa(
            X_train, y_train, k, 10/n, s0=s0, weights=class_weights,
            time_limit=exp_cfg.gurobi_timeout, verbose=exp_cfg.gurobi_verbose
        )
        y_score = X_test[:, support] @ beta + bias
        y_pred = np.sign(y_score)
        osp_perf = get_binary_cls_perf_metrics(y_test, y_pred, y_score)
        osp_results[(k, False)] = {**osp_perf, **model_stats}
        osp_results[(k, False)]['support'] = support
        osp_results[(k, False)]['beta'] = beta

        # Filtered
        s0 = np.zeros(exp_cfg.osp_heuristic_filter_size+1)
        filter_mask = np.isin(coef_filter, top_coefs[-k:]).tolist()
        filter_mask.append(False)  # For the bias term
        s0[filter_mask] = 1
        model_stats, support, beta, bias = sparse_classification_oa(
            X_train[:, coef_filter], y_train, k, 10/n, s0=s0, weights=class_weights,
            time_limit=exp_cfg.gurobi_timeout, verbose=exp_cfg.gurobi_verbose
        )
        y_score = X_test[:, coef_filter[support]] @ beta + bias
        y_pred = np.sign(y_score)
        osp_perf = get_binary_cls_perf_metrics(y_test, y_pred, y_score)
        osp_results[(k, True)] = {**osp_perf, **model_stats}
        osp_results[(k, True)]['support'] = support
        osp_results[(k, True)]['beta'] = beta

    layer_results = {
        'f_statistic': f_stat,
        'f_p_value': p_val,
        'mutual_info': mi,
        'lr': lr.coef_[0],
        'means': mean_dif,
        'osp_results': osp_results
    }
    return layer_results


def compare_feature_selection(exp_cfg, activation_dataset, feature_labels, filter=512):
    X_train, X_test, y_train, y_test = split_and_preprocess(
        exp_cfg, activation_dataset, feature_labels)

    n, d_act = X_train.shape

    selection_methods = ['f_stat', 'mi', 'lr', 'mean_dif', 'random']
    neuron_ranks = {}
    selection_times = {}
    for m in selection_methods:  # bigger is better for all
        start_t = time.time()
        neuron_ranks[m] = filtered_heuristic_neuron_ranking(
            X_train, y_train, m)
        selection_times[m] = time.time() - start_t

    ks = make_k_list(d_act, exp_cfg.max_k)
    layer_results = {}
    for k in ks:
        for m in selection_methods:
            support_indices = np.sort(neuron_ranks[m][-k:])
            k_lr = LogisticRegression(
                class_weight='balanced',
                penalty='l2',
                solver='saga',
                n_jobs=-1,
                max_iter=500
            )
            start_t = time.time()
            k_lr = k_lr.fit(X_train[:, support_indices], y_train)
            k_lr_elapsed_time = time.time() - start_t

            k_lr_score = k_lr.decision_function(X_test[:, support_indices])
            k_lr_pred = k_lr.predict(X_test[:, support_indices])

            layer_results[(m, k)] = get_binary_cls_perf_metrics(
                y_test, k_lr_pred, k_lr_score)
            layer_results[(m, k)]['elapsed_time'] = k_lr_elapsed_time
            layer_results[(m, k)]['support'] = support_indices
            layer_results[(m, k)]['coef'] = k_lr.coef_[0]

    results = {
        'selection_times': selection_times,
        'ranking': neuron_ranks,
        'method_scores': layer_results
    }
    return results


INNER_LOOP_FNS = {
    'dense_probe': dense_probe,
    'enumerate_monosemantic': enumerate_monosemantic,
    'optimal_sparse_probing': optimal_sparse_probing,
    'osp_tuning': osp_tuning,
    'osp_iterative_pruning': osp_iterative_pruning,
    'heuristic_sparsity_sweep': heuristic_sparsity_sweep,
    'fast_heuristic_sparsity_sweep': fast_heuristic_sparsity_sweep,
    'telescopic_sparsity_sweep': telescopic_sparsity_sweep,
    'rotation_baseline': rotation_baseline,
    'rotation_baseline_dxd': rotation_baseline_dxd,
    'iterative_pruning_rotation_baseline_dxd': iterative_pruning_rotation_baseline_dxd,
    'tune_telescoping_sparsity_sweep': tune_telescoping_sparsity_sweep,
    'test_osp_heuristic_filtering': test_osp_heuristic_filtering,
    'compare_feature_selection': compare_feature_selection,
    # regression probes
    'dense_regression_probe': dense_regression_probe,
    'heuristic_sparse_regression_sweep': heuristic_sparse_regression_sweep,
}
