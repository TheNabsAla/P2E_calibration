import numpy as np
import pandas as pd
from numpy.linalg import lstsq
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_openml
from tqdm import tqdm
import os


#-------- p-to-e conversion through P2E calibrator -------
from scipy.optimize import root_scalar
from scipy.optimize import minimize

# define numerically stable logistic 
def logistic(z):
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z, dtype=np.float64)

    pos = z >= 0
    neg = ~pos

    ez = np.exp(-z[pos])
    out[pos] = ez / (1.0 + ez)

    ez = np.exp(z[neg])
    out[neg] = 1.0 / (1.0 + ez)

    return out

# get C,s such that F is P2E calibrator
def get_C_s(alpha,n_calib):
    xk = np.array([k / (n_calib + 1) for k in range(1, n_calib + 2)])
    m = int(np.floor(alpha * (n_calib + 1)))
    x_mp1 = (m + 1) / (n_calib + 1)
    
    s = 0.45*alpha + x_mp1*0.55
    
    # Residual function
    def residual_F(C):
        if C == 0.0:
            mean_fk = 0.5
            f_alpha = 0.5
        else:
            fk = np.array([logistic(C * (x - s)) for x in xk])
            mean_fk = float(np.mean(fk))
            f_alpha = logistic(C * (alpha - s))
        return mean_fk - alpha * f_alpha
    
    # Solve for C via bisection
    res = root_scalar(residual_F, bracket=[1e-12, 1e6], method='brentq', xtol=1e-14)
    C = res.root
    return C,s

# Return P2E calibrator    
def f_p_to_e(x,alpha,C,s):
    f_alpha=logistic(C * (alpha - s))
    f=logistic(C * (x - s))
    return f/(alpha*f_alpha)


# ------------------------- P-value construction rules (Gasparin methods) -------------------------

# return min of cumulative sum of p-values
def f_e_avg(p):
    p = np.asarray(p)
    if p.size == 0:
        return np.nan
    mus = np.cumsum(p) / (np.arange(1, p.size + 1))
    return float(np.min(mus))

# Randomization : return mean p-values/(2-U)
def f_u_avg(p, u):
    p = np.asarray(p)
    return float(np.mean(p) / (2.0 - u))

# mininmum of f_e_avg and f_u_avg
def f_eu_avg(p, u):
    p = np.asarray(p)
    if p.size == 0:
        return np.nan
    a = p[0] / (2.0 - u)
    mus = np.cumsum(p) / (np.arange(1, p.size + 1))
    b = float(np.min(mus))
    return float(min(a, b))


# ------------------------- Prediction set construction  -------------------------

# Return (lower, upper) intervals where P_val > alpha
def set_cc(p, y_grid, alpha):

    p = np.asarray(p)
    y_grid = np.asarray(y_grid)
    if p.size != y_grid.size:
        raise ValueError("p and y_grid must have same length")

    mask = p > alpha
    if not np.any(mask):
        return None

    intervals = []
    i = 0
    n = mask.size
    while i < n:
        if mask[i]:
            start = i
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            lower = float(y_grid[start])
            upper = float(y_grid[j])
            intervals.append((lower, upper))
            i = j + 1
        else:
            i += 1

    if len(intervals) == 0:
        return None
    else:
        return np.array(intervals)

# Return (lower, upper) intervals where E_val < 1/alpha
def set_cc_eval(e, y_grid, alpha):
    e = np.asarray(e)
    y_grid = np.asarray(y_grid)
    if e.size != y_grid.size:
        raise ValueError("e and y_grid must have same length")

    mask = e < 1/alpha
    if not np.any(mask):
        return None

    # find runs of True values
    intervals = []
    i = 0
    n = mask.size
    while i < n:
        if mask[i]:
            start = i
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            # inclusive indices start..j
            lower = float(y_grid[start])
            upper = float(y_grid[j])
            intervals.append((lower, upper))
            i = j + 1
        else:
            i += 1

    if len(intervals) == 0:
        return None
    else:
        return np.array(intervals)

# ------------------------- Empirical coverage and lenght on test set -------------------------

def cov_int(intervals, y_test):
  
    if intervals is None:
        return 0
    intervals = np.asarray(intervals)
    if intervals.size == 0 or np.isnan(intervals).any():
        return 0
    contains = (intervals[:, 0] <= y_test) & (y_test <= intervals[:, 1])
    return int(np.sum(contains))

def len_int(intervals):
    if intervals is None:
        return 0.0
    intervals = np.asarray(intervals)
    if intervals.size == 0 or np.isnan(intervals).any():
        return 0.0
    lengths = intervals[:, 1] - intervals[:, 0]
    return float(np.sum(lengths))

# ------------------------- Models (OLS, RF, Lasso) -------------------------

def train_rf(X_train, y_train, ntree, max_features=1.0, random_state=None):
    X = np.asarray(X_train, dtype=float); y = np.asarray(y_train, dtype=float).ravel()
    model = RandomForestRegressor(n_estimators=ntree, n_jobs=-1,
                                  max_features=max_features, random_state=random_state)
    model.fit(X, y)
    return model

def predict_rf(model, x):
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)   
    preds = model.predict(x)
    return np.asarray(preds).ravel()

def ols_pseudo(X, y):
    # add_intercept=True to add intercept for OLS
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    
    X = np.column_stack([np.ones(X.shape[0]), X])
    beta, *_ = lstsq(X, y, rcond=None)
    return beta.ravel()

def ols_pseudo_predict(beta, x_test):
    beta = np.asarray(beta, dtype=float).ravel()
    x = np.asarray(x_test, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    
    x = np.column_stack([np.ones(x.shape[0]), x])
    return np.asarray(x @ beta).ravel()

def lasso_train(X, y, alpha=0.01, random_state=None):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    model = Lasso(alpha=alpha, max_iter=10000, random_state=random_state)
    model.fit(X, y)
    return model

def lasso_predict(model, X):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return np.asarray(model.predict(X)).ravel()


# ======================================================================
# ================== Cross-Conformal Prediction methods ================
# ======================================================================

# OLS
def cc_ols(y, X, x_test, K, alpha, n_grid=300, grid_factor=1.0, random_state=None):
    rng = np.random.default_rng(random_state)
    y = np.asarray(y).ravel()
    X = np.asarray(X)
    x_test = np.asarray(x_test)
    if x_test.ndim == 1:
        x_test = x_test.reshape(1, -1)

    n = y.size
    n_test = x_test.shape[0]
    max_abs_y = np.max(np.abs(y)) if y.size > 0 else 1.0
    grid = np.linspace(-grid_factor * max_abs_y, grid_factor * max_abs_y, num=n_grid)

    # Equal-size folds 
    m = n // K
    if m == 0:
        raise ValueError("n < K: cannot form equal-size folds.")
    used = m * K
    idx = rng.permutation(n)[:used]
    S_k = idx.reshape(K, m)          # K folds, each size m
    pool = idx                       # restrict training/calibration to trimmed pool

    p_vals = np.empty((n_grid, K, n_test), dtype=float)

    for k in range(K):
        idx_test = S_k[k]
        idx_train = np.setdiff1d(pool, idx_test)

        mu_hat   = ols_pseudo(X[idx_train, :], y[idx_train])
        pred_hat = np.asarray(ols_pseudo_predict(mu_hat, X[idx_test, :])).ravel()  # (m,)
        pred_new = np.asarray(ols_pseudo_predict(mu_hat, x_test)).ravel()          # (n_test,)

        diffs = np.abs(y[idx_test] - pred_hat)                                     # (m,)
        abs_errs_all = np.abs(grid[:, None] - pred_new[None, :])                   # (n_grid, n_test)
        cond = diffs[:, None, None] >= abs_errs_all[None, :, :]                    # (m, n_grid, n_test)
        p_vals[:, k, :] = (1.0 + cond.sum(axis=0)) / (m + 1.0)                     # (n_grid, n_test)

    pv_cc = p_vals.mean(axis=1)


    U_vals = rng.random(n_test)

    # P2E E-values methods:
    C, s = get_C_s(alpha, m)
    C2,s2 = get_C_s(2*alpha, m)
    E_mean = np.empty_like(pv_cc)
    E_exch = np.empty_like(pv_cc)
    E_exch_U = np.empty_like(pv_cc)
    E_mean_2 = np.empty_like(pv_cc)

    # ptoe calibrators of the literature
    E_mean_ind=np.empty_like(pv_cc)
    E_mean_log=np.empty_like(pv_cc)
    E_mean_pow=np.empty_like(pv_cc)
    E_mean_sqrt=np.empty_like(pv_cc)

    for j in range(n_test):
        u = U_vals[j]
        for i in range(n_grid):
            vals = p_vals[i, :, j]
            # p-to-e calibration (-indicator, log(p) and 2(1-p))
            evals_ind = (vals <= alpha).astype(float) / alpha
            evals_log = -np.log(vals)
            evals_pow = 5*(1-vals)**4
            evals_sqrt= vals**(-0.5) -1

            E_mean_ind[i, j] = np.mean(evals_ind) / u
            E_mean_log[i, j] = np.mean(evals_log) / u
            E_mean_pow[i, j] = np.mean(evals_pow) / u
            E_mean_sqrt[i, j]= np.mean(evals_sqrt)/u

            # P2E calibration
            e_vals = f_p_to_e(vals,alpha,C,s)
            cum_means = np.cumsum(e_vals) / np.arange(1, len(e_vals) + 1)

            E_mean[i, j] = np.mean(e_vals) / u
            E_exch[i, j] = np.max(cum_means)
            E_exch_U[i, j] = np.maximum(E_exch[i, j], e_vals[0] / u)

            # ECCP with 2alpha
            e_vals_2 = f_p_to_e(vals,2*alpha,C2,s2)
            E_mean_2[i, j] = np.mean(e_vals_2) / u

    pv_ecc = np.empty_like(pv_cc)
    for i in range(n_grid):
        for j in range(n_test):
            pv_ecc[i, j] = f_e_avg(p_vals[i, :, j])

    pv_ucc = pv_cc.copy()
    for l in range(n_test):
        u = U_vals[l]
        for i in range(n_grid):
            pv_ucc[i, l] = f_u_avg(p_vals[i, :, l], u)

    pv_eucc = pv_cc.copy()
    for l in range(n_test):
        u = U_vals[l]
        for i in range(n_grid):
            pv_eucc[i, l] = f_eu_avg(p_vals[i, :, l], u)

    pv_ccs = np.empty_like(pv_cc)
    for i in range(n_grid):
        for j in range(n_test):
            x = p_vals[i, :, j]
            pv_ccs[i, j] = (1.0 + np.sum(x * (m + 1.0) - 1.0)) / (used + 1.0)

    int_cc = [set_cc(pv_cc[:, j], grid, alpha) for j in range(n_test)] # 1-2α
    int_cce = [set_cc(pv_ecc[:, j], grid, alpha) for j in range(n_test)] # 1-2α
    int_ccu = [set_cc(pv_ucc[:, j], grid, alpha) for j in range(n_test)] # 1-2α
    int_cceu = [set_cc(pv_eucc[:, j], grid, alpha) for j in range(n_test)] # 1-2α
    int_ccs = [set_cc(pv_ccs[:, j], grid, alpha) for j in range(n_test)] # vovk  1-2α-2(1-α)(1-K/n)/(n/K+1) 
    int_cc_eval = [set_cc_eval(E_mean[:, j], grid, alpha) for j in range(n_test)] # 1-α
    int_cc_ev_exch = [set_cc_eval(E_exch[:, j], grid, alpha) for j in range(n_test)] # 1-α
    int_cc_ev_exch_U = [set_cc_eval(E_exch_U[:, j], grid, alpha) for j in range(n_test)] # 1-α
    int_cc_eval_2alpha = [set_cc_eval(E_mean_2[:, j], grid, 2*alpha) for j in range(n_test)] # 1-2α
    
    int_cc_eval_ind = [set_cc_eval(E_mean_ind[:, j], grid, alpha) for j in range(n_test)] # 1-α
    int_cc_eval_sqrt = [set_cc_eval(E_mean_sqrt[:, j], grid, alpha) for j in range(n_test)] # 1-α
    int_cc_eval_log = [set_cc_eval(E_mean_log[:, j], grid, alpha) for j in range(n_test)] # 1-α
    int_cc_eval_pow = [set_cc_eval(E_mean_pow[:, j], grid, alpha) for j in range(n_test)] # 1-α


    return {
        "p_vals": p_vals,
        "ys": grid,
        "int_cc": int_cc,
        "int_cce": int_cce,
        "int_ccu": int_ccu,
        "int_cceu": int_cceu,
        "int_ccs": int_ccs,
        "int_cc_eval": int_cc_eval,
        "int_cc_ev_exch": int_cc_ev_exch,
        "int_cc_ev_exch_U": int_cc_ev_exch_U,
        "int_cc_eval_2alpha":int_cc_eval_2alpha,
        "int_cc_eval_ind":int_cc_eval_ind,
        "int_cc_eval_sqrt":int_cc_eval_sqrt,
        "int_cc_eval_log":int_cc_eval_log,
        "int_cc_eval_pow":int_cc_eval_pow,
    }

# RF
def cc_rf(y, X, x_test, K, alpha, ntree, n_grid=300, grid_factor=1.0, random_state=None):
    y = np.asarray(y).ravel()
    X = np.asarray(X)
    x_test = np.asarray(x_test)
    if x_test.ndim == 1:
        x_test = x_test.reshape(1, -1)

    n = y.size
    n_test = x_test.shape[0]
    max_abs_y = np.max(np.abs(y)) if y.size > 0 else 1.0
    grid = np.linspace(-grid_factor * max_abs_y, grid_factor * max_abs_y, num=n_grid)

    rng = np.random.default_rng(random_state)
    # Equal-size folds 
    m = n // K
    if m == 0:
        raise ValueError("n < K: cannot form equal-size folds.")
    used = m * K
    idx = rng.permutation(n)[:used]
    S_k = idx.reshape(K, m)
    pool = idx

    p_vals = np.empty((n_grid, K, n_test), dtype=float)

    for k in range(K):
        idx_test = S_k[k]
        idx_train = np.setdiff1d(pool, idx_test)

        # --- RF here ---
        model = RandomForestRegressor(
            n_estimators=ntree, max_features=1.0, n_jobs=-1, random_state=random_state
        )
        model.fit(X[idx_train, :], y[idx_train])
        pred_hat = model.predict(X[idx_test, :])     # (m,)
        pred_new = model.predict(x_test)             # (n_test,)

        pred_hat = np.asarray(pred_hat).ravel()
        pred_new = np.asarray(pred_new).ravel()

        diffs = np.abs(y[idx_test] - pred_hat)                           # (m,)
        abs_errs_all = np.abs(grid[:, None] - pred_new[None, :])         # (n_grid, n_test)
        cond = diffs[:, None, None] >= abs_errs_all[None, :, :]          # (m, n_grid, n_test)
        p_vals[:, k, :] = (1.0 + cond.sum(axis=0)) / (m + 1.0)

    pv_cc = p_vals.mean(axis=1)

    U_vals = rng.random(n_test)

    # E-values methods:
    C, s = get_C_s(alpha, m)
    C2,s2 = get_C_s(2*alpha, m)

    E_mean = np.empty_like(pv_cc)
    E_exch = np.empty_like(pv_cc)
    E_exch_U = np.empty_like(pv_cc)
    E_mean_2 = np.empty_like(pv_cc)

    E_mean_ind=np.empty_like(pv_cc)
    E_mean_log=np.empty_like(pv_cc)
    E_mean_pow=np.empty_like(pv_cc)
    E_mean_sqrt=np.empty_like(pv_cc)
    

    for j in range(n_test):
        u = U_vals[j]
        for i in range(n_grid):
            vals = p_vals[i, :, j]
            
            # p-to-e calibration (-indicator, log(p) and 2(1-p))
            evals_ind = (vals <= alpha).astype(float) / alpha
            #evals_log = -np.log(vals)
            evals_log= -np.log(vals)
            evals_pow = 5*(1-vals)**4
            evals_sqrt=vals**(-0.5) -1

            E_mean_ind[i, j] = np.mean(evals_ind) / u
            E_mean_log[i, j] = np.mean(evals_log) / u
            E_mean_pow[i, j] = np.mean(evals_pow) / u
            E_mean_sqrt[i, j]= np.mean(evals_sqrt)/u

            # P2E
            e_vals = f_p_to_e(vals,alpha,C,s)
            cum_means = np.cumsum(e_vals) / np.arange(1, len(e_vals) + 1)

            E_mean[i, j] = np.mean(e_vals) / u
            E_exch[i, j] = np.max(cum_means)
            E_exch_U[i, j] = np.maximum(E_exch[i, j], e_vals[0] / u)

            # ECCP with 2alpha
            e_vals_2 = f_p_to_e(vals,2*alpha,C2,s2)
            E_mean_2[i, j] = np.mean(e_vals_2) / u

    pv_ecc = np.empty_like(pv_cc)
    for i in range(n_grid):
        for j in range(n_test):
            pv_ecc[i, j] = f_e_avg(p_vals[i, :, j])

    pv_ucc = pv_cc.copy()
    for l in range(n_test):
        u = U_vals[l]
        for i in range(n_grid):
            pv_ucc[i, l] = f_u_avg(p_vals[i, :, l], u)

    pv_eucc = pv_cc.copy()
    for l in range(n_test):
        u = U_vals[l]
        for i in range(n_grid):
            pv_eucc[i, l] = f_eu_avg(p_vals[i, :, l], u)

    pv_ccs = np.empty_like(pv_cc)
    for i in range(n_grid):
        for j in range(n_test):
            x = p_vals[i, :, j]
            pv_ccs[i, j] = (1.0 + np.sum(x * (m + 1.0) - 1.0)) / (used + 1.0)

    int_cc = [set_cc(pv_cc[:, j], grid, alpha) for j in range(n_test)] # 1-2α
    int_cce = [set_cc(pv_ecc[:, j], grid, alpha) for j in range(n_test)] # 1-2α
    int_ccu = [set_cc(pv_ucc[:, j], grid, alpha) for j in range(n_test)] # 1-2α
    int_cceu = [set_cc(pv_eucc[:, j], grid, alpha) for j in range(n_test)] # 1-2α
    int_ccs = [set_cc(pv_ccs[:, j], grid, alpha) for j in range(n_test)] # vovk  1-2α-2(1-α)(1-K/n)/(n/K+1) 
    int_cc_eval = [set_cc_eval(E_mean[:, j], grid, alpha) for j in range(n_test)] # 1-α
    int_cc_ev_exch = [set_cc_eval(E_exch[:, j], grid, alpha) for j in range(n_test)] # 1-α
    int_cc_ev_exch_U = [set_cc_eval(E_exch_U[:, j], grid, alpha) for j in range(n_test)] # 1-α
    int_cc_eval_2alpha = [set_cc_eval(E_mean_2[:, j], grid, 2*alpha) for j in range(n_test)] # 1-2α
    
    int_cc_eval_ind = [set_cc_eval(E_mean_ind[:, j], grid, alpha) for j in range(n_test)] # 1-α
    int_cc_eval_sqrt = [set_cc_eval(E_mean_sqrt[:, j], grid, alpha) for j in range(n_test)] # 1-α
    int_cc_eval_log = [set_cc_eval(E_mean_log[:, j], grid, alpha) for j in range(n_test)] # 1-α

    int_cc_eval_pow = [set_cc_eval(E_mean_pow[:, j], grid, alpha) for j in range(n_test)] # 1-α
    return {
        "p_vals": p_vals,
        "ys": grid,
        "int_cc": int_cc,
        "int_cce": int_cce,
        "int_ccu": int_ccu,
        "int_cceu": int_cceu,
        "int_ccs": int_ccs,
        "int_cc_eval": int_cc_eval,
        "int_cc_ev_exch": int_cc_ev_exch,
        "int_cc_ev_exch_U": int_cc_ev_exch_U,
        "int_cc_eval_2alpha":int_cc_eval_2alpha,
        "int_cc_eval_ind":int_cc_eval_ind,
        "int_cc_eval_sqrt":int_cc_eval_sqrt,
        "int_cc_eval_log":int_cc_eval_log,
        "int_cc_eval_pow":int_cc_eval_pow,
    }


# Lasso
def cc_lasso(y, X, x_test, K, alpha, lambda_=1.0, n_grid=300, grid_factor=1.0, random_state=None): 
    y = np.asarray(y).ravel()
    X = np.asarray(X)
    x_test = np.asarray(x_test)
    if x_test.ndim == 1:
        x_test = x_test.reshape(1, -1)

    n = y.size
    n_test = x_test.shape[0]
    max_abs_y = np.max(np.abs(y)) if y.size > 0 else 1.0
    grid = np.linspace(-grid_factor * max_abs_y, grid_factor * max_abs_y, num=n_grid)

    rng = np.random.default_rng(random_state)
    # Equal-size folds 
    m = n // K
    if m == 0:
        raise ValueError("n < K: cannot form equal-size folds.")
    used = m * K
    idx = rng.permutation(n)[:used]
    S_k = idx.reshape(K, m)
    pool = idx

    p_vals = np.empty((n_grid, K, n_test), dtype=float)

    for k in range(K):
        idx_test = S_k[k]
        idx_train = np.setdiff1d(pool, idx_test)

        model = Lasso(alpha=lambda_, fit_intercept=True, max_iter=10000, random_state=random_state)
        model.fit(X[idx_train, :], y[idx_train])
        pred_hat = model.predict(X[idx_test, :])   # (m,)
        pred_new = model.predict(x_test)           # (n_test,)

        pred_hat = np.asarray(pred_hat).ravel()
        pred_new = np.asarray(pred_new).ravel()

        diffs = np.abs(y[idx_test] - pred_hat)                           # (m,)
        abs_errs_all = np.abs(grid[:, None] - pred_new[None, :])         # (n_grid, n_test)
        cond = diffs[:, None, None] >= abs_errs_all[None, :, :]          # (m, n_grid, n_test)
        p_vals[:, k, :] = (1.0 + cond.sum(axis=0)) / (m + 1.0)

    pv_cc = p_vals.mean(axis=1)


    U_vals = rng.random(n_test)

    
    C, s = get_C_s(alpha, m) # for α e-value (E<1/α)
    C2,s2 = get_C_s(2*alpha, m) # for 2α e-value (E<1/2α)

    E_mean = np.empty_like(pv_cc)
    E_exch = np.empty_like(pv_cc)
    E_exch_U = np.empty_like(pv_cc)
    E_mean_2 = np.empty_like(pv_cc)

    E_mean_ind=np.empty_like(pv_cc)
    E_mean_log=np.empty_like(pv_cc)
    E_mean_pow=np.empty_like(pv_cc)
    E_mean_sqrt=np.empty_like(pv_cc)

    for j in range(n_test):
        u = U_vals[j]
        for i in range(n_grid):
            vals = p_vals[i, :, j]
            
            # p-to-e calibration (-indicator, log(p) and 2(1-p))
            evals_ind = (vals <= alpha).astype(float) / alpha
            evals_log = -np.log(vals)
            evals_pow = 5*(1-vals)**4
            evals_sqrt= vals**(-0.5) -1

            E_mean_ind[i, j] = np.mean(evals_ind) / u
            E_mean_log[i, j] = np.mean(evals_log) / u
            E_mean_pow[i, j] = np.mean(evals_pow) / u
            E_mean_sqrt[i, j]= np.mean(evals_sqrt)/u

            # P2E calibration
            e_vals = f_p_to_e(vals,alpha,C,s)
            cum_means = np.cumsum(e_vals) / np.arange(1, len(e_vals) + 1)

            E_mean[i, j] = np.mean(e_vals) / u
            E_exch[i, j] = np.max(cum_means)
            E_exch_U[i, j] = np.maximum(E_exch[i, j], e_vals[0] / u)

            # ECCP with 2α
            e_vals_2 = f_p_to_e(vals,2*alpha,C2,s2)
            E_mean_2[i, j] = np.mean(e_vals_2) / u


    pv_ecc = np.empty_like(pv_cc)
    for i in range(n_grid):
        for j in range(n_test):
            pv_ecc[i, j] = f_e_avg(p_vals[i, :, j])

    pv_ucc = pv_cc.copy()
    for l in range(n_test):
        u = U_vals[l]
        for i in range(n_grid):
            pv_ucc[i, l] = f_u_avg(p_vals[i, :, l], u)

    pv_eucc = pv_cc.copy()
    for l in range(n_test):
        u = U_vals[l]
        for i in range(n_grid):
            pv_eucc[i, l] = f_eu_avg(p_vals[i, :, l], u)

    pv_ccs = np.empty_like(pv_cc)
    for i in range(n_grid):
        for j in range(n_test):
            x = p_vals[i, :, j]
            pv_ccs[i, j] = (1.0 + np.sum(x * (m + 1.0) - 1.0)) / (used + 1.0)

    int_cc = [set_cc(pv_cc[:, j], grid, alpha) for j in range(n_test)] # 1-2α
    int_cce = [set_cc(pv_ecc[:, j], grid, alpha) for j in range(n_test)] # 1-2α
    int_ccu = [set_cc(pv_ucc[:, j], grid, alpha) for j in range(n_test)] # 1-2α
    int_cceu = [set_cc(pv_eucc[:, j], grid, alpha) for j in range(n_test)] # 1-2α
    int_ccs = [set_cc(pv_ccs[:, j], grid, alpha) for j in range(n_test)] # vovk  1-2α-2(1-α)(1-K/n)/(n/K+1) 
    int_cc_eval = [set_cc_eval(E_mean[:, j], grid, alpha) for j in range(n_test)] # 1-α
    int_cc_ev_exch = [set_cc_eval(E_exch[:, j], grid, alpha) for j in range(n_test)] # 1-α
    int_cc_ev_exch_U = [set_cc_eval(E_exch_U[:, j], grid, alpha) for j in range(n_test)] # 1-α
    int_cc_eval_2alpha = [set_cc_eval(E_mean_2[:, j], grid, 2*alpha) for j in range(n_test)] # 1-2α

    
    int_cc_eval_ind = [set_cc_eval(E_mean_ind[:, j], grid, alpha) for j in range(n_test)] # 1-α
    int_cc_eval_sqrt = [set_cc_eval(E_mean_sqrt[:, j], grid, alpha) for j in range(n_test)] # 1-α
    int_cc_eval_log = [set_cc_eval(E_mean_log[:, j], grid, alpha) for j in range(n_test)] # 1-α
    int_cc_eval_pow = [set_cc_eval(E_mean_pow[:, j], grid, alpha) for j in range(n_test)] # 1-α

    return {
        "p_vals": p_vals,
        "ys": grid,
        "int_cc": int_cc,
        "int_cce": int_cce,
        "int_ccu": int_ccu,
        "int_cceu": int_cceu,
        "int_ccs": int_ccs,
        "int_cc_eval": int_cc_eval,
        "int_cc_ev_exch": int_cc_ev_exch,
        "int_cc_ev_exch_U": int_cc_ev_exch_U,
        "int_cc_eval_2alpha":int_cc_eval_2alpha,
        "int_cc_eval_ind":int_cc_eval_ind,
        "int_cc_eval_sqrt":int_cc_eval_sqrt,
        "int_cc_eval_log":int_cc_eval_log,
        "int_cc_eval_pow":int_cc_eval_pow,
    }


