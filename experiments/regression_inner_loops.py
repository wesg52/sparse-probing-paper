import time
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from .regression_probes import *
from .metrics import get_regression_perf_metrics


def make_regression_k_list():
    base_ks = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 14]
    exp_ks = list((2 ** np.linspace(4, 8, 13)).astype(int))
    return base_ks + exp_ks


def dense_regression_probe(exp_cfg, activation_dataset, regression_target):
    """
    Train a dense probe on the activation dataset.

    Parameters
    ----------
    exp_cfg : as specified by the CLI in probing_experiment.py
    activation_dataset : np.ndarray (n_samples, n_neurons)
    regression_target : np.ndarray (n_samples) with regression targets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        activation_dataset, regression_target,
        test_size=exp_cfg.test_set_frac, random_state=exp_cfg.seed)

    lr = ElasticNet(precompute=True)

    start_t = time.time()
    lr = lr.fit(X_train, y_train)
    elapsed_time = time.time() - start_t 
    lr_pred = lr.predict(X_test)

    results = get_regression_perf_metrics(y_test, lr_pred)
    results['elapsed_time'] = elapsed_time
    results['n_iter'] = lr.n_iter_
    results['coef'] = lr.coef_
    return results


def heuristic_sparse_regression_sweep(exp_cfg, activation_dataset, regression_target):
    """
    Train a heuristic sparse probe on the activation dataset for varying k.

    Parameters
    ----------
    exp_cfg : as specified by the CLI in probing_experiment.py
    activation_dataset : np.ndarray (n_samples, n_neurons)
    regression_target : np.ndarray (n_samples) with regression targets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        activation_dataset, regression_target,
        test_size=exp_cfg.test_set_frac, random_state=exp_cfg.seed)

    neuron_ranking = get_heuristic_neuron_ranking_regression(
        X_train, y_train, 'f_stat')

    layer_results = {}
    for k in make_regression_k_list()[::-1]:
        support = np.sort(neuron_ranking[-k:])
        lr = ElasticNet(precompute=True)
        start_t = time.time()
        lr = lr.fit(X_train[:, support], y_train)
        elapsed_time = time.time() - start_t

        lr_pred = lr.predict(X_test[:, support])
        layer_results[k] = get_regression_perf_metrics(y_test, lr_pred)
        layer_results[k]['elapsed_time'] = elapsed_time
        layer_results[k]['n_iter'] = lr.n_iter_
        layer_results[k]['coef'] = lr.coef_
        layer_results[k]['support'] = support

        # rerank according to the linear regression coefficients
        neuron_ranking = np.zeros(len(neuron_ranking))
        neuron_ranking[support] = np.abs(lr.coef_)
        neuron_ranking = np.argsort(neuron_ranking)

    return layer_results


def optimal_sparse_regression_probe(exp_cfg, activation_dataset, regression_target):
    """
    Train a sparse probe on the activation dataset.

    Parameters
    ----------
    exp_cfg : as specified by the CLI in probing_experiment.py
    activation_dataset : np.ndarray (n_samples, n_neurons)
    regression_target : np.ndarray (n_samples) with regression targets.
    """
    raise NotImplementedError
