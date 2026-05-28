import numpy as np
from itertools import product

from utils import conformal_quantile, get_p_to_e


# ============================================================
# Small utilities
# ============================================================

def _models(models):
    return list(models.values()) if isinstance(models, dict) else list(models)


def _rows(A, idx):
    return A.iloc[idx] if hasattr(A, "iloc") else A[idx]


def _idxs(cal_idx_by_model, K, n):
    if cal_idx_by_model is None:
        return [np.arange(n) for _ in range(K)]

    if len(cal_idx_by_model) != K:
        raise ValueError(f"Expected {K} calibration index sets, got {len(cal_idx_by_model)}.")

    out = []
    for k, idx in enumerate(cal_idx_by_model):
        idx = np.asarray(idx, dtype=int).ravel()

        if len(idx) == 0:
            raise ValueError(f"Model {k} has empty calibration set.")

        if idx.min() < 0 or idx.max() >= n:
            raise ValueError(f"Model {k} has calibration indices outside [0, {n}).")

        out.append(idx)

    return out


def _cal_scores(models, X_calib, y_calib, cal_idx_by_model):
    models = _models(models)
    K = len(models)
    idxs = _idxs(cal_idx_by_model, K, len(y_calib))

    scores = []

    for k, model in enumerate(models):
        idx = idxs[k]
        Xk = _rows(X_calib, idx)
        yk = np.asarray(_rows(y_calib, idx)).ravel()
        pk = np.asarray(model.predict(Xk)).ravel()

        scores.append(np.abs(yk - pk))

    return scores


def _pvals(scores, cal_scores):
    scores = np.asarray(scores)

    if scores.ndim == 1:
        return np.array([
            (1.0 + np.sum(cal_scores[k] >= scores[k])) / (len(cal_scores[k]) + 1.0)
            for k in range(len(cal_scores))
        ])

    if scores.ndim == 2:
        M, K = scores.shape
        out = np.zeros((M, K))

        for k in range(K):
            out[:, k] = (
                1.0
                + (cal_scores[k][None, :] >= scores[:, k][:, None]).sum(axis=1)
            ) / (len(cal_scores[k]) + 1.0)

        return out

    raise ValueError("scores must be 1D or 2D.")


def _evals(pvals, p_to_e_fns):
    pvals = np.asarray(pvals)

    if pvals.ndim == 1:
        return np.array([
            np.asarray(p_to_e_fns[k](np.array([pvals[k]]))).ravel()[0]
            for k in range(len(p_to_e_fns))
        ])

    if pvals.ndim == 2:
        out = np.zeros_like(pvals, dtype=float)

        for k, fn in enumerate(p_to_e_fns):
            out[:, k] = np.asarray(fn(pvals[:, k])).ravel()

        return out

    raise ValueError("pvals must be 1D or 2D.")




def _preds(models, X):
    return np.stack([np.asarray(m.predict(X)).ravel() for m in models], axis=1)


####### Conformal Aggregation methods #######

# ============================================================
# CM / CR
# ============================================================

def Majority_vote(
    models,
    X_calib,
    y_calib,
    cal_idx_by_model,
    X_test,
    y_test,
    alpha,
    M,
    U_test=None,
    Random=False,
):
    models = _models(models)

    if Random and U_test is None:
        raise ValueError("U_test is required when Random=True.")

    y_test = np.asarray(y_test).ravel()

    cal_scores = _cal_scores(models, X_calib, y_calib, cal_idx_by_model)
    preds_test = _preds(models, X_test)

    qs = np.array([
        conformal_quantile(cal_scores[k], alpha / 2)
        for k in range(len(models))
    ])

    lowers = preds_test - qs[None, :]
    uppers = preds_test + qs[None, :]

    sets, lengths, covered = [], [], []

    for i in range(len(X_test)):
        threshold = 0.5 + U_test[i] / 2 if Random else 0.5

        inside_true = (lowers[i] <= y_test[i]) & (y_test[i] <= uppers[i])
        covered.append(inside_true.mean() > threshold)

    

        finite = np.isfinite(lowers[i]) & np.isfinite(uppers[i])

        lo = np.min(lowers[i, finite])
        hi = np.max(uppers[i, finite])

        y_grid = np.linspace(lo, hi, M)
        step = y_grid[1] - y_grid[0] if M > 1 else 0.0

        inside = (
            (lowers[i, :, None] <= y_grid[None, :])
            & (y_grid[None, :] <= uppers[i, :, None])
        )

        mask = inside.mean(axis=0) > threshold

        sets.append(y_grid[mask])
        lengths.append(mask.sum() * step)

    return sets, np.array(lengths), float(np.mean(covered)), float(np.mean(lengths))


# ============================================================
# P-value aggregation
# ============================================================

def Pvalue_aggregation(
    models,
    X_calib,
    y_calib,
    cal_idx_by_model,
    X_test,
    y_test,
    alpha,
    M,
):
    models = _models(models)
    y_test = np.asarray(y_test).ravel()

    cal_scores = _cal_scores(models, X_calib, y_calib, cal_idx_by_model)
    max_scores = np.array([s.max() for s in cal_scores])

    preds_test = _preds(models, X_test)

    sets, lengths, covered = [], [], []


    for i in range(len(X_test)):
        true_scores = np.abs(y_test[i] - preds_test[i])
        true_pvals = _pvals(true_scores, cal_scores)

        covered.append(2.0 * true_pvals.mean() > alpha)

        lo = np.min(preds_test[i] - 2*max_scores) # Wider grid used for P_Agg which tends to be conservative
        hi = np.max(preds_test[i] + 2*max_scores)

        y_grid = np.linspace(lo, hi, M)
        step = y_grid[1] - y_grid[0] if M > 1 else 0.0

        grid_scores = np.abs(y_grid[:, None] - preds_test[i][None, :])
        pvals = _pvals(grid_scores, cal_scores)

        mask = 2.0 * pvals.mean(axis=1) > alpha

        sets.append(y_grid[mask])
        lengths.append(mask.sum() * step)

    return sets, np.array(lengths), float(np.mean(covered)), float(np.mean(lengths))


# ============================================================
# ECA / UR-ECA
# ============================================================

def Evalue_aggregation(
    models,
    X_calib,
    y_calib,
    cal_idx_by_model,
    X_test,
    y_test,
    alpha,
    U_test,
    M,
    P_TO_E=None,
    Random=False,
):
    models = _models(models)

    if Random and U_test is None:
        raise ValueError("U_test is required when Random=True.")

    y_test = np.asarray(y_test).ravel()

    cal_scores = _cal_scores(models, X_calib, y_calib, cal_idx_by_model)
    max_scores = np.array([s.max() for s in cal_scores])

    preds_test = _preds(models, X_test)

    p_to_e_fns = [
        get_p_to_e(P_TO_E, alpha, len(s))
        for s in cal_scores
    ]

    sets, lengths, covered = [], [], []

    for i in range(len(X_test)):
        threshold = U_test[i] / alpha if Random else 1.0 / alpha

        true_scores = np.abs(y_test[i] - preds_test[i])
        true_pvals = _pvals(true_scores, cal_scores)
        true_evals = _evals(true_pvals, p_to_e_fns)

        covered.append(true_evals.mean() < threshold)

        lo = np.min(preds_test[i] - 3*max_scores)
        hi = np.max(preds_test[i] + 3*max_scores)
      
        if P_TO_E in ["log", "linear", "sqrt"]:
            lo = np.min(preds_test[i] - 8*max_scores)
            hi = np.max(preds_test[i] + 8*max_scores)

        y_grid = np.linspace(lo, hi, M)
        step = y_grid[1] - y_grid[0] if M > 1 else 0.0

        grid_scores = np.abs(y_grid[:, None] - preds_test[i][None, :])
        pvals = _pvals(grid_scores, cal_scores)
        evals = _evals(pvals, p_to_e_fns)

        mask = evals.mean(axis=1) < threshold

        sets.append(y_grid[mask])
        lengths.append(mask.sum() * step)

    return sets, np.array(lengths), float(np.mean(covered)), float(np.mean(lengths))


# ============================================================
# COLA-S (from https://arxiv.org/pdf/2511.12065)
# ============================================================

def COLA_S(
    models,
    X_calib,
    y_calib,
    cal_idx_by_model,
    X_test,
    y_test,
    alpha,
    G,
    seed,
):
    models = _models(models)
    K = len(models)
    n = len(y_calib)
    y_test = np.asarray(y_test).ravel()

    base_idxs = _idxs(cal_idx_by_model, K, n)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)

    i1, i2 = idx[: n // 2], idx[n // 2 :]

    idxs_1 = [np.intersect1d(base_idxs[k], i1) for k in range(K)]
    idxs_2 = [np.intersect1d(base_idxs[k], i2) for k in range(K)]

    cal_scores_1 = _cal_scores(models, X_calib, y_calib, idxs_1)
    cal_scores_2 = _cal_scores(models, X_calib, y_calib, idxs_2)

    P1 = _preds(models, _rows(X_calib, i1))
    P_test = _preds(models, X_test)

    vals = np.linspace(0, alpha, G)
    allocations = np.array([
        a for a in product(vals, repeat=K)
        if sum(a) <= alpha
    ])

    best_a = None
    best_L = np.inf

    for a in allocations:
        qs = np.array([
            conformal_quantile(cal_scores_1[k], a[k])
            for k in range(K)
        ])

        lower = np.max(P1 - qs[None, :], axis=1)
        upper = np.min(P1 + qs[None, :], axis=1)

        L = np.maximum(0.0, upper - lower).mean()

        if L < best_L:
            best_L = L
            best_a = a

    qs = np.array([
        conformal_quantile(cal_scores_2[k], best_a[k])
        for k in range(K)
    ])

    lower = np.max(P_test - qs[None, :], axis=1)
    upper = np.min(P_test + qs[None, :], axis=1)

    lengths = np.maximum(0.0, upper - lower)
    covered = (lower <= y_test) & (y_test <= upper)

    return (
        lower,
        upper,
        lengths,
        float(np.mean(covered)),
        float(np.mean(lengths)),
        best_a,
    )


# ============================================================
# WECA / UR-WECA
# ============================================================
def sample_weights(K, B=500, seed=42):
    rng = np.random.default_rng(seed)

    W = rng.dirichlet(np.ones(K), size=B)

    # Include one-hot weights: [1,0,0,...], [0,1,0,...], etc.
    one_hot = np.eye(K)

    # Include uniform weights as a baseline
    uniform = np.ones(K) / K

    return np.vstack([W, one_hot, uniform])

def Evalue_aggregation_weighted(
    models,
    X_calib,
    y_calib,
    cal_idx_by_model,
    X_test,
    y_test,
    alpha,
    U_test,
    M,
    seed,
    P_TO_E=None,
    B=100,
    Random=False,
):
    models = _models(models)
    K = len(models)
    n = len(y_calib)

    if Random and U_test is None:
        raise ValueError("U_test is required when Random=True.")

    y_test = np.asarray(y_test).ravel()

    base_idxs = _idxs(cal_idx_by_model, K, n)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
 
    n1 = n // 4
    n2 = n // 4

    i1 = idx[:n1]
    i2 = idx[n1:n1+n2]
    i3 = idx[n1+n2:]
    #i1, i2, i3 = np.array_split(idx, 3)

    idxs_1 = [np.intersect1d(base_idxs[k], i1) for k in range(K)]
    idxs_3 = [np.intersect1d(base_idxs[k], i3) for k in range(K)]

    cal_scores_1 = _cal_scores(models, X_calib, y_calib, idxs_1)
    cal_scores_3 = _cal_scores(models, X_calib, y_calib, idxs_3)

    max_scores_1 = np.array([s.max() for s in cal_scores_1])
    max_scores_3 = np.array([s.max() for s in cal_scores_3])

    p_to_e_1 = [get_p_to_e(P_TO_E, alpha, len(s)) for s in cal_scores_1]
    p_to_e_3 = [get_p_to_e(P_TO_E, alpha, len(s)) for s in cal_scores_3]

    # ---------- choose weights on split 2 ----------
    X2 = _rows(X_calib, i2)
    preds_2 = _preds(models, X2)

    W = sample_weights(K, B=B, seed=seed)
    avg_lengths = np.zeros(len(W))


    for t, w in enumerate(W):

        lens = []

        for i in range(len(X2)):
            lo = np.min(preds_2[i] - 3*max_scores_1)
            hi = np.max(preds_2[i] + 3*max_scores_1)
            if P_TO_E in ["log", "linear", "sqrt"]: # wider grid for alternatives calibrators which are conservative
                lo = np.min(preds_2[i] - 8*max_scores_1)
                hi = np.max(preds_2[i] + 8*max_scores_1)
            

            y_grid = np.linspace(lo, hi, M)
            step = y_grid[1] - y_grid[0] if M > 1 else 0.0

            grid_scores = np.abs(y_grid[:, None] - preds_2[i][None, :])
            pvals = _pvals(grid_scores, cal_scores_1)
            evals = _evals(pvals, p_to_e_1)

            mask = evals @ w < 1.0 / alpha
            lens.append(mask.sum() * step)

        avg_lengths[t] = np.mean(lens)

    w_star =  W[np.argmin(avg_lengths)]

    # ---------- final test using split 3 ----------
    preds_test = _preds(models, X_test)

    sets, lengths, covered = [], [], []

    for i in range(len(X_test)):
        threshold = U_test[i] / alpha if Random else 1.0 / alpha

        true_scores = np.abs(y_test[i] - preds_test[i])
        true_pvals = _pvals(true_scores, cal_scores_3)
        true_evals = _evals(true_pvals, p_to_e_3)

        covered.append(true_evals @ w_star < threshold)

        lo = np.min(preds_test[i] - 3*max_scores_3)
        hi = np.max(preds_test[i] + 3*max_scores_3)
        if P_TO_E in ["log", "linear", "sqrt"]:
            lo = np.min(preds_test[i] - 8*max_scores_3)
            hi = np.max(preds_test[i] + 8*max_scores_3)

        y_grid = np.linspace(lo, hi, M)
        step = y_grid[1] - y_grid[0] if M > 1 else 0.0

        grid_scores = np.abs(y_grid[:, None] - preds_test[i][None, :])
        pvals = _pvals(grid_scores, cal_scores_3)
        evals = _evals(pvals, p_to_e_3)

        mask = evals @ w_star < threshold

        sets.append(y_grid[mask])
        lengths.append(mask.sum() * step)

    return (
        sets,
        np.array(lengths),
        float(np.mean(covered)),
        float(np.mean(lengths)),
        w_star,
    )