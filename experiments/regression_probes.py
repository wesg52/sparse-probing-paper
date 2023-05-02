import gurobipy as gp
import numpy as np
from sklearn.feature_selection import f_regression, mutual_info_regression, r_regression
from sklearn.linear_model import LinearRegression, Lasso


def solve_inner_problem(X, Y, s, gamma):
    indices = np.where(s > 0.5)[0]
    n, d = X.shape
    denom = 2*n
    Xs = X[:, indices]

    alpha = Y - Xs @ (np.linalg.inv(np.eye(len(indices)) /
                      gamma + Xs.T @ Xs) @ (Xs.T @ Y))
    obj = np.dot(Y, alpha) / denom
    tmp = X.T @ alpha
    grad = -gamma * tmp**2 / denom
    return obj, grad


def sparse_regression_oa(X, Y, k, gamma, s0, time_limit=60, verbose=True):
    n, d = X.shape

    gp_env = gp.Env()  # need env for cluster
    model = gp.Model("classifier", env=gp_env)

    s = model.addVars(d, vtype=gp.GRB.BINARY, name="support")
    t = model.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name="objective")

    model.addConstr(gp.quicksum(s) <= k, name="l0")

    if len(s0) == 0:
        s0 = np.zeros(d)
        s0[range(int(k))] = 1

    obj0, grad0 = solve_inner_problem(X, Y, s0, gamma)
    model.addConstr(
        t >= obj0 + gp.quicksum(grad0[i] * (s[i] - s0[i]) for i in range(d)))
    model.setObjective(t, gp.GRB.MINIMIZE)

    def outer_approximation(model, where):
        if where == gp.GRB.Callback.MIPSOL:
            s_bar = model.cbGetSolution(model._vars)
            s_vals = np.array([a for a in s_bar.values()])
            obj, grad = solve_inner_problem(X, Y, s_vals, gamma)
            model.cbLazy(
                t >= obj + gp.quicksum(grad[i] * (s[i] - s_vals[i]) for i in range(d)))

    model._vars = s
    model.params.OutputFlag = 1 if verbose else 0
    model.Params.lazyConstraints = 1
    model.Params.timeLimit = time_limit
    model.optimize(outer_approximation)

    support_indices = sorted([i for i in range(len(s)) if s[i].X > 0.5])

    X_s = X[:, support_indices]
    beta = np.zeros(d)
    sol = np.linalg.solve(np.eye(int(k)) / gamma + X_s.T @ X_s, X_s.T @ Y)
    beta[support_indices] = gamma * X_s.T @ (Y - X_s @ sol)

    model_stats = {
        'obj': model.ObjVal,
        'obj_bound': model.ObjBound,
        'mip_gap': model.MIPGap,
        'model_status': model.Status,
        'sol_count': model.SolCount,
        'iter_count': model.IterCount,
        'node_count': model.NodeCount,
        'runtime': model.Runtime
    }

    model.dispose()
    gp_env.dispose()

    return model_stats, beta, support_indices


def get_heuristic_neuron_ranking_regression(X, y, method):
    if method == 'l1':
        lr = Lasso()
        lr = lr.fit(X, y)
        ranks = np.argsort(np.abs(lr.coef_[0]))
    elif method == 'f_stat':
        f_stat, p_val = f_regression(X, y)
        ranks = np.argsort(f_stat)
    elif method == 'mi':
        mi = mutual_info_regression(X, y)
        ranks = np.argsort(mi)

    elif method == 'correlation':
        corr = r_regression(X, y)
        ranks = np.argsort(np.abs(corr))
    else:
        raise ValueError('Invalid method')
    return ranks
