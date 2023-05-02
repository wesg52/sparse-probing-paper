import math
import random
import numpy as np
import os
import pickle
from sklearn import svm
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import *


def gelu(x):
    return x / (1 + np.exp(-1.702 * x))


def fake_gelu_dataset(n, d, k, mean_shift=1, imbalance=0.5):
    x = np.random.normal(0, 1, size=(n, d))
    true_support = np.random.choice(d, size=k, replace=False)
    true_classes = np.random.choice(n, size=int(n*imbalance), replace=False)
    y = np.zeros(n)
    y[true_classes] = 1
    y = y * 2 - 1  # to {+1, -1}

    x[:, true_support] += y[:, None] * mean_shift
    activations = gelu(x)
    return activations, y, true_support


def log_uniform(a, b):
    return 10**random.uniform(math.log10(a), math.log10(b))


def score_features(feature_scores, s_set):
    k = len(s_set)
    top_features = set(np.argsort(feature_scores)[-k:])
    return len(top_features.intersection(s_set)) / k


def feature_selection_results(X, y, s):
    pos_class = (y * 2 - 1).astype(bool)
    mean_dif = X[pos_class, :].mean(axis=0) - X[~pos_class, :].mean(axis=0)

    mi = mutual_info_classif(X, y)

    f_stat, p_val = f_classif(X, y)

    lr = LogisticRegression(class_weight='balanced',
                            penalty='l2', solver='saga', n_jobs=-1)
    lr = lr.fit(X, y)

    svm = LinearSVC(loss='hinge')
    svm = svm.fit(X, y)

    s_set = set(s)

    coeff_acc = {
        'mean_dif': score_features(np.abs(mean_dif), s_set),
        'mi': score_features(mi, s_set),
        'f_stat': score_features(f_stat, s_set),
        'lr_mag': score_features(np.abs(lr.coef_[0]), s_set),
        'svm_mag': score_features(np.abs(svm.coef_[0]), s_set),
    }
    return coeff_acc


def run_synthetic_data_experiment(n_trials):
    results = {}
    for i in range(n_trials):
        n = int(log_uniform(500, 30_000))
        d = random.choice(50 * 2**np.arange(3, 8))
        k = np.random.choice(np.arange(2, 20))
        mean_shift = log_uniform(0.05, 5)
        class_imbalance = log_uniform(0.01, 0.5)

        X, y, s = fake_gelu_dataset(
            n, d, k, mean_shift=mean_shift, imbalance=class_imbalance)

        trial_results = feature_selection_results(X, y, s)
        results[i] = trial_results
    return results


if __name__ == '__main__':
    EXPERIMENT_NAME = 'synthetic_data_feature_selection_comparison'
    N_TRIALS = 200

    seed = int(os.getenv('SLURM_ARRAY_TASK_ID', 1))
    random.seed(seed)
    np.random.seed(seed)

    save_path = os.path.join(
        os.getenv('RESULTS_DIR', 'results'), EXPERIMENT_NAME)
    os.makedirs(save_path, exist_ok=True)
    file_name = f'shard_{seed}.p'

    results = run_synthetic_data_experiment(N_TRIALS)
    pickle.dump(results, open(os.path.join(save_path, file_name), 'wb'))
