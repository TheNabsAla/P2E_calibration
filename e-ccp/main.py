'''
Generate results of different CCP methods and save in csv files 
'''
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
import os
from eccp_utils import *
import time as ts

from data_loader import load_dataset


if __name__ == "__main__":

    DATASET = "boston"   # "boston", "abalone", "parkinson"
    
    
    B = 100  # number of seeds    
    alpha   = 0.1  

    X, y, cfg = load_dataset(DATASET)
    n       = y.shape[0]
    n_train = cfg["n_train"]
    n_test  = cfg["n_test"] if cfg["n_test"] is not None else (n - n_train)
    K       = 5 # K = 5, 10,... 30
    lambda_ = cfg["lambda_"]
    ntree   = cfg["ntree"]
    print(f"K = {K} for {B} seeds (dataset={cfg['name']}, n={n}, n_train={n_train}, n_test={n_test})")

    #OLS
    cov_ols_fn = []
    len_ols_fn = []
    cov_res_ols = np.full((B, 13), np.nan)
    len_res_ols = np.full((B,13), np.nan)
    #RF
    cov_rf_fn = []
    len_rf_fn = []
    cov_res_rf = np.full((B, 13), np.nan)
    len_res_rf = np.full((B, 13), np.nan)
    #Lasso
    cov_lasso_fn = []
    len_lasso_fn = []
    cov_res_lasso = np.full((B, 13), np.nan)
    len_res_lasso = np.full((B, 13), np.nan)


    rng = np.random.default_rng(1234)

    for b in tqdm(range(B)):
        seed = b + 45
        rng_b = np.random.default_rng(seed)

        all_idx  = np.arange(n)
        train_idx = rng_b.choice(all_idx, size=n_train, replace=False)
        test_idx  = np.setdiff1d(all_idx, train_idx)

        ytrain = y[train_idx]
        Xtrain = X[train_idx, :]
        ytest  = y[test_idx]
        Xtest  = X[test_idx, :]

        def train_fun_rf(X_tr, y_tr, ntree=ntree, seed=seed):
            return train_rf(X_tr, y_tr, ntree=ntree, max_features=1.0, random_state=seed)

        def train_fun_lasso(X_tr, y_tr, lam=lambda_, seed=seed):
            return lasso_train(X_tr, y_tr, alpha=lam, random_state=seed)

        # ========================= OLS =========================
        cr_ols = cc_ols(
            y=ytrain, X=Xtrain, x_test=Xtest, K=K, alpha=alpha,
            n_grid=300, grid_factor=1.0, random_state=seed
        )
  
        cov_ols = np.full((n_test, 13), np.nan)
        len_ols = np.full((n_test, 13), np.nan)

        # ========================= RF ==========================
        cr_rf = cc_rf(
            y=ytrain, X=Xtrain, x_test=Xtest, K=K, alpha=alpha,ntree=ntree,
            n_grid=300, grid_factor=1.0, random_state=seed
        )
 
        cov_rf = np.full((n_test, 13), np.nan)
        len_rf = np.full((n_test, 13), np.nan)

        # ======================== LASSO ========================
        cr_lasso = cc_lasso(
            y=ytrain, X=Xtrain, x_test=Xtest, K=K, alpha=alpha,
            n_grid=300, grid_factor=1.0, random_state=seed, lambda_=lambda_
        )

        cov_lasso = np.full((n_test, 13), np.nan)
        len_lasso = np.full((n_test, 13), np.nan)

        for i in range(n_test):
            #OLS------------------------------------------------
            cov_ols[i, 0] = cov_int(cr_ols["int_cc"][i], ytest[i])
            len_ols[i, 0] = len_int(cr_ols["int_cc"][i])
            # e-cc
            cov_ols[i, 1] = cov_int(cr_ols["int_cce"][i], ytest[i])
            len_ols[i, 1] = len_int(cr_ols["int_cce"][i])
            # u-cc
            cov_ols[i, 2] = cov_int(cr_ols["int_ccu"][i], ytest[i])
            len_ols[i, 2] = len_int(cr_ols["int_ccu"][i])
            # eu-cc
            cov_ols[i, 3] = cov_int(cr_ols["int_cceu"][i], ytest[i])
            len_ols[i, 3] = len_int(cr_ols["int_cceu"][i])
            # cc (Vovk)
            cov_ols[i, 4] = cov_int(cr_ols["int_ccs"][i], ytest[i])
            len_ols[i, 4] = len_int(cr_ols["int_ccs"][i])
            # ECCP
            cov_ols[i, 5] = cov_int(cr_ols['int_cc_eval'][i], ytest[i])
            len_ols[i, 5] = len_int(cr_ols['int_cc_eval'][i])
            # ECCP-Exch
            cov_ols[i, 6] = cov_int(cr_ols['int_cc_ev_exch'][i], ytest[i])
            len_ols[i, 6] = len_int(cr_ols['int_cc_ev_exch'][i])
            # UR-ECCP-Exch
            cov_ols[i, 7] = cov_int(cr_ols['int_cc_ev_exch_U'][i], ytest[i])
            len_ols[i, 7] = len_int(cr_ols['int_cc_ev_exch_U'][i])

            cov_ols[i, 8] = cov_int(cr_ols['int_cc_eval_ind'][i], ytest[i])
            len_ols[i, 8] = len_int(cr_ols['int_cc_eval_ind'][i])
            cov_ols[i, 9] = cov_int(cr_ols['int_cc_eval_sqrt'][i], ytest[i])
            len_ols[i, 9] = len_int(cr_ols['int_cc_eval_sqrt'][i])
            cov_ols[i, 10] = cov_int(cr_ols['int_cc_eval_log'][i], ytest[i])
            len_ols[i, 10] = len_int(cr_ols['int_cc_eval_log'][i])
            cov_ols[i, 11] = cov_int(cr_ols['int_cc_eval_pow'][i], ytest[i])
            len_ols[i, 11] = len_int(cr_ols['int_cc_eval_pow'][i])
            # ECCP (2alpha)
            cov_ols[i, 12] = cov_int(cr_ols['int_cc_eval_2alpha'][i], ytest[i])
            len_ols[i, 12] = len_int(cr_ols['int_cc_eval_2alpha'][i])
            #---------------------------------------------
            #RF-------------------------------------------
            cov_rf[i, 0] = cov_int(cr_rf["int_cc"][i], ytest[i])
            len_rf[i, 0] = len_int(cr_rf["int_cc"][i])
            # e-cc
            cov_rf[i, 1] = cov_int(cr_rf["int_cce"][i], ytest[i])
            len_rf[i, 1] = len_int(cr_rf["int_cce"][i])
            # u-cc
            cov_rf[i, 2] = cov_int(cr_rf["int_ccu"][i], ytest[i])
            len_rf[i, 2] = len_int(cr_rf["int_ccu"][i])
            # eu-cc
            cov_rf[i, 3] = cov_int(cr_rf["int_cceu"][i], ytest[i])
            len_rf[i, 3] = len_int(cr_rf["int_cceu"][i])
            # cc (Vovk)
            cov_rf[i, 4] = cov_int(cr_rf["int_ccs"][i], ytest[i])
            len_rf[i, 4] = len_int(cr_rf["int_ccs"][i])
            # ECCP
            cov_rf[i, 5] = cov_int(cr_rf['int_cc_eval'][i], ytest[i])
            len_rf[i, 5] = len_int(cr_rf['int_cc_eval'][i])
            # ECCP-Exch
            cov_rf[i, 6] = cov_int(cr_rf['int_cc_ev_exch'][i], ytest[i])
            len_rf[i, 6] = len_int(cr_rf['int_cc_ev_exch'][i])
            # UR-ECCP-Exch
            cov_rf[i, 7] = cov_int(cr_rf['int_cc_ev_exch_U'][i], ytest[i])
            len_rf[i, 7] = len_int(cr_rf['int_cc_ev_exch_U'][i])

            cov_rf[i, 8] = cov_int(cr_rf['int_cc_eval_ind'][i], ytest[i])
            len_rf[i, 8] = len_int(cr_rf['int_cc_eval_ind'][i])
            cov_rf[i, 9] = cov_int(cr_rf['int_cc_eval_sqrt'][i], ytest[i])
            len_rf[i, 9] = len_int(cr_rf['int_cc_eval_sqrt'][i])
            cov_rf[i, 10] = cov_int(cr_rf['int_cc_eval_log'][i], ytest[i])
            len_rf[i, 10] = len_int(cr_rf['int_cc_eval_log'][i])
            cov_rf[i, 11] = cov_int(cr_rf['int_cc_eval_pow'][i], ytest[i])
            len_rf[i, 11] = len_int(cr_rf['int_cc_eval_pow'][i])

            # ECCP (2alpha)
            cov_rf[i, 12] = cov_int(cr_rf['int_cc_eval_2alpha'][i], ytest[i])
            len_rf[i, 12] = len_int(cr_rf['int_cc_eval_2alpha'][i])

            #---------------------------------------------
            #Lasso-------------------------------------------
            # mod-cc
            cov_lasso[i, 0] = cov_int(cr_lasso["int_cc"][i], ytest[i])
            len_lasso[i, 0] = len_int(cr_lasso["int_cc"][i])
            # e-cc
            cov_lasso[i, 1] = cov_int(cr_lasso["int_cce"][i], ytest[i])
            len_lasso[i, 1] = len_int(cr_lasso["int_cce"][i])
            # u-cc
            cov_lasso[i, 2] = cov_int(cr_lasso["int_ccu"][i], ytest[i])
            len_lasso[i, 2] = len_int(cr_lasso["int_ccu"][i])
            # eu-cc
            cov_lasso[i, 3] = cov_int(cr_lasso["int_cceu"][i], ytest[i])
            len_lasso[i, 3] = len_int(cr_lasso["int_cceu"][i])
            # cc (Vovk)
            cov_lasso[i, 4] = cov_int(cr_lasso["int_ccs"][i], ytest[i])
            len_lasso[i, 4] = len_int(cr_lasso["int_ccs"][i])
            # ECCP
            cov_lasso[i, 5] = cov_int(cr_lasso['int_cc_eval'][i], ytest[i])
            len_lasso[i, 5] = len_int(cr_lasso['int_cc_eval'][i])
            # ECCP-Exch
            cov_lasso[i, 6] = cov_int(cr_lasso['int_cc_ev_exch'][i], ytest[i])
            len_lasso[i, 6] = len_int(cr_lasso['int_cc_ev_exch'][i])
            # UR-ECCP-Exch
            cov_lasso[i, 7] = cov_int(cr_lasso['int_cc_ev_exch_U'][i], ytest[i])
            len_lasso[i, 7] = len_int(cr_lasso['int_cc_ev_exch_U'][i])

            cov_lasso[i, 8] = cov_int(cr_lasso['int_cc_eval_ind'][i], ytest[i])
            len_lasso[i, 8] = len_int(cr_lasso['int_cc_eval_ind'][i])
            cov_lasso[i, 9] = cov_int(cr_lasso['int_cc_eval_sqrt'][i], ytest[i])
            len_lasso[i, 9] = len_int(cr_lasso['int_cc_eval_sqrt'][i])
            cov_lasso[i, 10] = cov_int(cr_lasso['int_cc_eval_log'][i], ytest[i])
            len_lasso[i, 10] = len_int(cr_lasso['int_cc_eval_log'][i])
            cov_lasso[i, 11] = cov_int(cr_lasso['int_cc_eval_pow'][i], ytest[i])
            len_lasso[i, 11] = len_int(cr_lasso['int_cc_eval_pow'][i])
          # ECCP (2alpha)
            cov_lasso[i, 12] = cov_int(cr_lasso['int_cc_eval_2alpha'][i], ytest[i])
            len_lasso[i, 12] = len_int(cr_lasso['int_cc_eval_2alpha'][i])
        #OLS
        cov_ols_fn.append(cov_ols)
        len_ols_fn.append(len_ols)
        cov_res_ols[b, :] = np.nanmean(cov_ols, axis=0)
        len_res_ols[b, :] = np.nanmean(len_ols, axis=0)
        #RF
        cov_rf_fn.append(cov_rf)
        len_rf_fn.append(len_rf)
        cov_res_rf[b, :] = np.nanmean(cov_rf, axis=0)
        len_res_rf[b, :] = np.nanmean(len_rf, axis=0)
        #Lasso
        cov_lasso_fn.append(cov_lasso)
        len_lasso_fn.append(len_lasso)
        cov_res_lasso[b, :] = np.nanmean(cov_lasso, axis=0)
        len_res_lasso[b, :] = np.nanmean(len_lasso, axis=0)

        print("Iteration", b + 1,flush=True)

methods = [
    "mod-cross", "e-mod-cross", "u-mod-cross", "eu-mod-cross",
    "cross", "ECCP", "ECCP_exch", "UR-ECCP_exch",
    "ECCP(ind)", "ECCP(sqrt)", "ECCP(log)", "ECCP(pow)", "ECCP (2α)"
]

base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_name = cfg["name"]

results_dir = os.path.join(base_dir, "Results")
os.makedirs(results_dir, exist_ok=True)


def to_long(model, cov_res, len_res):
    rows = []
    B, M = cov_res.shape

    if M != len(methods):
        raise ValueError(f"Expected {len(methods)} methods, got {M}.")

    for s in range(B):
        for j in range(M):
            rows.append([
                model,
                s,
                methods[j],
                float(cov_res[s, j]),
                float(len_res[s, j]),
            ])

    return pd.DataFrame(
        rows,
        columns=["model", "seed", "method", "coverage", "length"]
    )


df_seed = pd.concat([
    to_long("OLS",   cov_res_ols,   len_res_ols),
    to_long("RF",    cov_res_rf,    len_res_rf),
    to_long("Lasso", cov_res_lasso, len_res_lasso),
], ignore_index=True)

seed_csv = os.path.join(results_dir, f"seedwise_{dataset_name}_K={K}_{B}seeds.csv")
df_seed.to_csv(seed_csv, index=False)
print(f"Saved seed-wise results to {seed_csv}")


def make_block(cov_res, len_res, label):
    _, M = cov_res.shape

    if M != len(methods):
        raise ValueError(f"Expected {len(methods)} methods, got {M}.")

    return pd.DataFrame(
        np.vstack([
            np.nanmean(len_res, axis=0),
            np.nanstd(len_res, axis=0),
            np.nanmean(cov_res, axis=0),
            np.nanstd(cov_res, axis=0),
        ]),
        index=[
            f"{label}_Mean",
            f"{label}_std_mean",
            f"{label}_Coverage",
            f"{label}_std_cov",
        ],
        columns=methods,
    )


df_tab = pd.concat([
    make_block(cov_res_ols, len_res_ols, "OLS"),
    make_block(cov_res_rf, len_res_rf, "RF"),
    make_block(cov_res_lasso, len_res_lasso, "Lasso"),
])

csv_out_file = os.path.join(results_dir, f"{dataset_name}_K={K}_{B}seeds.csv")
df_tab.to_csv(csv_out_file, float_format="%.3f")
print(f"Saved results table to {csv_out_file}")