'''   
Plot figures of Cross-CP vs ECCP evolution with number trees of RF regressor (unstable for few trees)
'''

import numpy as np
import pandas as pd
import os
from tqdm import tqdm


from eccp_utils import cc_rf, cov_int, len_int
from data_loader import load_dataset

def run_experiment(
    DATASET="boston",
    K=5,
    alpha=0.1,
    B=20,
    ntree_list=(3, 6, 9, 12, 15, 19, 23, 26, 29, 32, 36, 39, 42, 45, 48, 50),
):
    X, y, cfg = load_dataset(DATASET)
    n       = y.shape[0]
    n_train = cfg["n_train"]
    n_test  = cfg["n_test"] if cfg["n_test"] is not None else (n - n_train)

    METHOD_KEYS = {
        "cross":         "int_ccs",
        "e-CCP_mean":    "int_cc_eval"
    }
    method_names = list(METHOD_KEYS.keys())

    rows = []
    for nt in tqdm(ntree_list):
        # per-seed aggregates 
        seed_cov_means = {m: [] for m in method_names}
        seed_len_means = {m: [] for m in method_names}

        for b in range(B):
            seed  = 45 + b
            rng_b = np.random.default_rng(seed)

            all_idx   = np.arange(n)
            train_idx = rng_b.choice(all_idx, size=n_train, replace=False)
            test_idx  = np.setdiff1d(all_idx, train_idx)

            Xtrain, ytrain = X[train_idx, :], y[train_idx]
            Xtest,  ytest  = X[test_idx, :],  y[test_idx]

            cr_rf = cc_rf(
                y=ytrain, X=Xtrain, x_test=Xtest,
                K=K, alpha=alpha,
                ntree=nt, n_grid=300, grid_factor=1.0,
                random_state=seed
            )

            cov_per_method = {m: [] for m in method_names}
            len_per_method = {m: [] for m in method_names}

            for i in range(n_test):
                for m, key in METHOD_KEYS.items():
                    intervals = cr_rf[key][i]
                    cov_per_method[m].append(cov_int(intervals, ytest[i]))
                    len_per_method[m].append(len_int(intervals))

            # store seed-level means
            for m in method_names:
                seed_cov_means[m].append(float(np.mean(cov_per_method[m])))
                seed_len_means[m].append(float(np.mean(len_per_method[m])))

        # summarize across seeds for this ntree
        row = {"ntree": nt}
        for m in method_names:
            cov_mean = float(np.mean(seed_cov_means[m]))
            cov_sd   = float(np.std(seed_cov_means[m], ddof=1)) if len(seed_cov_means[m]) > 1 else 0.0
            len_mean = float(np.mean(seed_len_means[m]))
            len_sd   = float(np.std(seed_len_means[m], ddof=1)) if len(seed_len_means[m]) > 1 else 0.0

         
            row[f"{m}_coverage"]     = cov_mean
            row[f"{m}_coverage_sd"]  = cov_sd
            row[f"{m}_length"]       = len_mean
            row[f"{m}_length_sd"]    = len_sd

        rows.append(row)
        print(f"Done ntree={nt}", flush=True)

    df = pd.DataFrame(rows).sort_values("ntree").reset_index(drop=True)
    return df


DATASET = "boston" # choose dataset
df = run_experiment(DATASET=DATASET, K=10, alpha=0.1, B=25)
base_dir = os.getcwd()
results_dir = os.path.join(base_dir, "Results_plots")
os.makedirs(results_dir, exist_ok=True)

csv_path = os.path.join(results_dir, f"rf_ccp_vs_ntree_{DATASET}.csv")
df.to_csv(csv_path, index=False, float_format="%.6f")
print(f"Saved results table to {csv_path}")



import pandas as pd
import matplotlib.pyplot as plt

results_dir = "Results_plots"
in_csv = os.path.join(results_dir, f"rf_ccp_vs_ntree_{DATASET}.csv")

DATASET = "boston"

df = pd.read_csv(in_csv)
print("Loaded:", in_csv)
print("Columns:", list(df.columns))

# auto-detect method names (present as both mean & length)
cov_means = [c for c in df.columns if c.endswith("_coverage")]
len_means = [c for c in df.columns if c.endswith("_length") and not c.endswith("_length_sd")]
methods = sorted(set(c.replace("_coverage", "") for c in cov_means)
                 .intersection(c.replace("_length", "") for c in len_means))

if not methods:
    raise ValueError("No matching method columns found.")

def sd_col(base, suffix):
    c = f"{base}_{suffix}_sd"
    return c if c in df.columns else None

# ---- Coverage plot (mean ± std) ----
plt.figure()
for m in methods:
    mean_col = f"{m}_coverage"
    std_col  = sd_col(m, "coverage")
    y = df[mean_col].values
    plt.plot(df["ntree"], y, label=m)
    if std_col is not None:
        s = df[std_col].values
        plt.fill_between(df["ntree"], y - s, y + s, alpha=0.2)

plt.xlabel("Number of trees (ntree)")
plt.ylabel("Mean coverage (± sd over seeds)")
plt.ylim(0.7, 1.0)
plt.title(f"RF CCP/e-CCP coverage vs ntree ({DATASET})")
plt.axhline(0.9, linestyle="--")
plt.legend()
cov_png = os.path.join(results_dir, f"rf_ccp_vs_ntree_{DATASET}_coverage.png")
plt.savefig(cov_png, bbox_inches="tight", dpi=300)
print("Saved:", cov_png)

# ---- Length plot (mean ± std) ----
plt.figure()
for m in methods:
    mean_col = f"{m}_length"
    std_col  = sd_col(m, "length")
    y = df[mean_col].values
    plt.plot(df["ntree"], y, label=m)
    if std_col is not None:
        s = df[std_col].values
        plt.fill_between(df["ntree"], y - s, y + s, alpha=0.2)

plt.xlabel("Number of trees (ntree)")
plt.ylabel("Mean set length (± sd over seeds)")
plt.title(f"RF CCP/e-CCP length vs ntree ({DATASET})")
plt.legend()

len_png = os.path.join(results_dir, f"rf_ccp_vs_ntree_{DATASET}_length.png")
plt.savefig(len_png, bbox_inches="tight", dpi=300)
print("Saved:", len_png)
