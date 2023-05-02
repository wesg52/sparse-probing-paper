from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import f_classif, mutual_info_classif
import gurobipy as gp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math


def generate_gradient_cut(X, y, s, reg, weights):
    indices = s > 0.5

    X_sub = X[:, indices]
    n, k = X_sub.shape

    sub_env = gp.Env()  # need env for cluster
    model = gp.Model("subproblem", env=sub_env)
    model.params.OutputFlag = 0

    alpha = model.addMVar(n, vtype=gp.GRB.CONTINUOUS,
                          lb=-weights, ub=weights, name='dual_vars')
    model.addConstr(alpha.sum() == 0)
    # 1-norm SVM conjugate constraint
    model.addConstr(y * alpha / weights <= 0)
    model.addConstr(-1 <= y * alpha / weights)

    XTa = model.addMVar(k, lb=-gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)
    model.addConstr(XTa == X_sub.T @ alpha)

    model.setObjective(
        -(y.T @ alpha) - (reg / 2) * XTa @ XTa,
        gp.GRB.MAXIMIZE
    )
    model.optimize()

    obj = model.objVal
    alpha_vals = alpha.X
    grad = -(reg / 2) * (X.T @ alpha_vals) ** 2

    model.dispose()
    sub_env.dispose()

    return obj, grad, alpha_vals


def generate_gradient_cut_gurobi95(X, y, s, reg, weights=1):
    indices = s > 0.5

    X_sub = X[:, indices]
    n, k = X_sub.shape

    sub_env = gp.Env()  # need env for cluster
    model = gp.Model("subproblem", env=sub_env)
    model.params.OutputFlag = 0

    alpha = model.addVars(n, vtype=gp.GRB.CONTINUOUS,
                          lb=-1, ub=1, name='dual_vars')
    model.addConstr(gp.quicksum(alpha) == 0)
    # 1-norm SVM conjugate constraint
    model.addConstrs((y[i] * alpha[i] / weights[i] <= 0 for i in range(n)))
    model.addConstrs((-1 <= y[i] * alpha[i] / weights[i] for i in range(n)))

    XTa = model.addVars(k, lb=-gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)
    model.addConstrs((XTa[i] == gp.quicksum(X_sub[j, i] * alpha[j]
                     for j in range(n)) for i in range(k)))

    model.setObjective(
        -gp.quicksum(y[i] * alpha[i]
                     for i in range(n)) - (reg / 2) * gp.quicksum(XTa[i]**2 for i in range(k)),
        gp.GRB.MAXIMIZE
    )
    model.optimize()

    obj = model.objVal
    alpha_vals = np.array([alpha[i].X for i in range(n)])
    grad = -(reg / 2) * (X.T @ alpha_vals) ** 2

    model.dispose()
    sub_env.dispose()

    return obj, grad, alpha_vals


def sparse_classification_oa(X, Y, k, reg, s0, weights=1, time_limit=60, verbose=True):
    n, d = X.shape

    if isinstance(weights, int) and weights == 1:
        weights = np.ones(n)

    gp_env = gp.Env()  # need env for cluster
    model = gp.Model("classifier", env=gp_env)

    s = model.addVars(d, vtype=gp.GRB.BINARY, name="support")
    t = model.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name="objective")

    model.addConstr(gp.quicksum(s) == k, name="l0")
    # model.addConstr(s[d-1] == 1)

    obj0, grad0, alpha0 = generate_gradient_cut(
        X, Y, s0, reg, weights)
    model.addConstr(
        t >= obj0 + gp.quicksum(grad0[i] * (s[i] - s0[i]) for i in range(d)))
    model.setObjective(t, gp.GRB.MINIMIZE)

    alpha_opt = alpha0
    n_cuts_added = 0

    def outer_approximation(model, where):
        if where == gp.GRB.Callback.MIPSOL:
            s_bar = model.cbGetSolution(model._vars)
            s_vals = np.array([a for a in s_bar.values()])

            obj, grad, alpha = generate_gradient_cut(
                X, Y, s_vals, reg, weights)
            nonlocal alpha_opt
            alpha_opt = alpha
            nonlocal n_cuts_added
            n_cuts_added += 1

            model.cbLazy(
                t >= obj + gp.quicksum(grad[i] * (s[i] - s_vals[i])
                                       for i in range(d))
            )

    model._vars = s
    model.params.OutputFlag = 1 if verbose else 0
    model.Params.lazyConstraints = 1
    model.Params.timeLimit = time_limit
    model.optimize(outer_approximation)

    support_indices = sorted([i for i in range(len(s)) if s[i].X > 0.5])
    beta_opt = -reg * X[:, support_indices].T @ alpha_opt

    margin = (np.abs(alpha_opt) > 1e-6)
    bias = ((Y - X[:, support_indices] @ beta_opt) * weights)[margin].mean()

    model_stats = {
        'obj': model.ObjVal,
        'obj_bound': model.ObjBound,
        'mip_gap': model.MIPGap,
        'model_status': model.Status,
        'sol_count': model.SolCount,
        'iter_count': model.IterCount,
        'node_count': model.NodeCount,
        'n_cuts': n_cuts_added,
        'runtime': model.Runtime
    }

    model.dispose()
    gp_env.dispose()

    # return bias separately
    return model_stats, support_indices, beta_opt, bias


def get_balanced_class_weights(y):
    frac_true = (y == 1).mean()
    frac_false = 1 - frac_true

    weights = np.zeros(len(y))
    weights[y == 1] = 1 / (2 * frac_true)
    weights[~(y == 1)] = 1 / (2 * frac_false)
    return weights


def get_heuristic_neuron_ranking(X, y, method):
    pos_class = y == 1
    if method == 'lr':
        lr = LogisticRegression(
            class_weight='balanced', C=0.1,
            penalty='l1', solver='saga', n_jobs=-1)
        lr = lr.fit(X, y)
        ranks = np.argsort(np.abs(lr.coef_[0]))
    elif method == 'svm':
        svm = LinearSVC(loss='hinge')
        svm = svm.fit(X, y)
        ranks = np.argsort(np.abs(svm.coef_))
    elif method == 'f_stat':
        f_stat, p_val = f_classif(X, y)
        ranks = np.argsort(f_stat)
    elif method == 'mi':
        mi = mutual_info_classif(X, y)
        ranks = np.argsort(mi)

    elif method == 'mean_dif':
        mean_dif = np.abs(X[pos_class].mean(axis=0) -
                          X[~pos_class].mean(axis=0))
        ranks = np.argsort(mean_dif)
    elif method == 'random':
        ranks = np.random.permutation(X.shape[1])
    else:
        raise ValueError('Invalid method')
    return ranks


def filtered_heuristic_neuron_ranking(X, y, method, filter_size=512):
    pos_class = y == 1
    mean_dif = np.abs(X[pos_class].mean(axis=0) -
                      X[~pos_class].mean(axis=0))
    ranks = np.argsort(mean_dif)
    filtered_subset = np.sort(ranks[-filter_size:])
    X_filtered = X[:, filtered_subset]
    top_filtered_ranks = get_heuristic_neuron_ranking(X_filtered, y, method)

    return filtered_subset[top_filtered_ranks]


def heuristic_feature_selection_binary_cls(X, y, method, k):
    if method == 'svm':
        svm = LinearSVC(loss='hinge')
        svm = svm.fit(X, y)
        return np.argsort(np.abs(svm.coef_))[-k:]
    elif method == 'lr':
        lr = LogisticRegression(class_weight='balanced',
                                penalty='l2', solver='saga', n_jobs=-1)
        lr = lr.fit(X, y)
        return np.argsort(np.abs(lr.coef_[0]))[-k:]

    elif method == 'f_stat':
        f_stat, p_val = f_classif(X, y)
        return np.argsort(f_stat)[-k:]

    elif method == 'mi':
        mi = mutual_info_classif(X, y)
        return np.argsort(mi)[-k:]

    elif method == 'means':
        pos_class = y == 1
        mean_dif = X[pos_class, :].mean(axis=0) - X[~pos_class, :].mean(axis=0)
        return np.argsort(np.argmax(mean_dif))[-k:]

    else:
        raise ValueError('Invalid method for heuristic feature selection')


def heuristic_probe():
    pass
