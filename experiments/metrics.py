import numpy as np
from sklearn.metrics import *


def downsample_perf_curves(curve, pts_to_keep=100):
    n = len(curve)
    if n <= pts_to_keep:
        return curve
    else:
        idx = np.round(np.linspace(0, n - 1, pts_to_keep)).astype(int)
        return curve[idx]


def get_binary_cls_perf_metrics(y_test, y_pred, y_score):
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    fowlkes_mallows_index = (precision_score(
        y_test, y_pred) * recall_score(y_test, y_pred))**0.5
    classifier_results = {
        'test_mcc': matthews_corrcoef(y_test, y_pred),
        'test_cohen_kappa': cohen_kappa_score(y_test, y_pred),
        'test_fmi': fowlkes_mallows_index,
        'test_f1_score': f1_score(y_test, y_pred),
        'test_f0.5_score': fbeta_score(y_test, y_pred, beta=0.5),
        'test_f2_score': fbeta_score(y_test, y_pred, beta=2),
        'test_pr_auc': auc(recall, precision),
        'test_acc': accuracy_score(y_test, y_pred),
        'test_balanced_acc': balanced_accuracy_score(y_test, y_pred),
        'test_precision': precision_score(y_test, y_pred),
        'test_recall': recall_score(y_test, y_pred),
        'test_average_precision': average_precision_score(y_test, y_pred),
        'test_roc_auc': roc_auc_score(y_test, y_score),
        'test_precision_curve': downsample_perf_curves(precision),
        'test_recall_curve': downsample_perf_curves(recall),
    }
    return classifier_results


def get_regression_perf_metrics(y_test, y_pred):
    return {
        'explained_variance': explained_variance_score(y_test, y_pred),
        'max_error': max_error(y_test, y_pred),
        'mean_absolute_error': mean_absolute_error(y_test, y_pred),
        'mean_squared_error': mean_squared_error(y_test, y_pred),
        'median_absolute_error': median_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'mean_absolute_percentage_error': mean_absolute_percentage_error(
            y_test, y_pred),
        'd2_absolute_error': d2_absolute_error_score(y_test, y_pred),
        'd2_pinball_score': d2_pinball_score(y_test, y_pred),
        'd2_tweedie_score': d2_tweedie_score(y_test, y_pred),
    }
