from pathlib import Path
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import *
from config import *
from methods import *


P_TO_E_LIST = ["log", "linear", "sqrt", "AoN", "P2E"]



def serialize_extra(x):
    """
    Make arrays/lists safe to save in CSV.
    """
    if x is None:
        return ""

    if isinstance(x, np.ndarray):
        return json.dumps(x.tolist())

    if isinstance(x, (list, tuple)):
        return json.dumps(list(x))

    return str(x)



def run_one_dataset(dataset_name, dataset_config, seeds, alpha, M, B):
    X, Y = load_dataset(dataset_config)
    X, Y = X[:data_limit], Y[:data_limit]

    all_results = []

    for seed in tqdm(seeds, desc=f"Dataset: {dataset_name}"):
        print(f"\nRunning dataset = {dataset_name}, seed = {seed}")

        X_train, X_cal, y_train, y_cal, X_test, y_test = split_data(X, Y, seed)

        models = make_models(seed)

        fitted_models = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            fitted_models[name] = model

        fitted_model_list = list(fitted_models.values())

        calib_sets_idx = [
            random_subsample(X_cal, y_cal, seed=seed + i)
            for i in range(len(fitted_model_list))
        ]
        print("Per-model calibration sizes:", [len(s) for s in calib_sets_idx])

        # Shared randomness for all randomized methods for this dataset/seed.
        U_test = np.random.default_rng(seed).uniform(0, 1, size=len(X_test))

        results = []

        # -----------------------------
        # Majority Vote: CM
        # -----------------------------
        majority_sets, majority_lengths, majority_cov, majority_avg_len = Majority_vote(
        fitted_model_list, X_cal, y_cal, calib_sets_idx,
        X_test, y_test, alpha=alpha, M=M,
        U_test=U_test, Random=False)
        

        results.append({
            "Dataset": dataset_name,
            "Seed": seed,
            "Method": "CM",
            "P_TO_E": "",
            "Random": False,
            "Coverage": majority_cov,
            "Avg Length": majority_avg_len,
            "Extra": "",
        })

        # -----------------------------
        # Randomized Majority Vote: CR
        # -----------------------------
        rand_sets, rand_lengths, rand_cov, rand_avg_len = Majority_vote(
        fitted_model_list, X_cal, y_cal, calib_sets_idx,
        X_test, y_test, alpha=alpha, M=M,
        U_test=U_test, Random=True)

        results.append({
            "Dataset": dataset_name,
            "Seed": seed,
            "Method": "CR",
            "P_TO_E": "",
            "Random": True,
            "Coverage": rand_cov,
            "Avg Length": rand_avg_len,
            "Extra": "",
        })

        # -----------------------------
        # P-value Aggregation
        # -----------------------------
        p_sets, p_lengths, p_cov, p_avg_len = Pvalue_aggregation(
        fitted_model_list, X_cal, y_cal, calib_sets_idx,
        X_test, y_test, alpha=alpha, M=M)

        results.append({
            "Dataset": dataset_name,
            "Seed": seed,
            "Method": "P-value Aggregation",
            "P_TO_E": "",
            "Random": False,
            "Coverage": p_cov,
            "Avg Length": p_avg_len,
            "Extra": "",
        })

        # -----------------------------
        # E-value Aggregation:
        # ECA(calibrator), UR-ECA(calibrator)
        # -----------------------------
        for p_to_e in P_TO_E_LIST:
            for random_flag in [False, True]:
                e_sets, e_lengths, e_cov, e_avg_len = Evalue_aggregation(
                fitted_model_list, X_cal, y_cal, calib_sets_idx,
                X_test, y_test, alpha=alpha, U_test=U_test,
                M=M, P_TO_E=p_to_e, Random=random_flag
                )

                method_name = f"UR-ECA({p_to_e})" if random_flag else f"ECA({p_to_e})"

                results.append({
                    "Dataset": dataset_name,
                    "Seed": seed,
                    "Method": method_name,
                    "P_TO_E": p_to_e,
                    "Random": random_flag,
                    "Coverage": e_cov,
                    "Avg Length": e_avg_len,
                    "Extra": "",
                })

        # -----------------------------
        # COLA-S
        # -----------------------------
        cola_lower, cola_upper, cola_lengths, cola_cov, cola_avg_len, best_a = COLA_S(
        fitted_model_list, X_cal, y_cal, calib_sets_idx,
        X_test, y_test, alpha=alpha, G=G, seed=seed)

        results.append({
            "Dataset": dataset_name,
            "Seed": seed,
            "Method": "COLA-S",
            "P_TO_E": "",
            "Random": False,
            "Coverage": cola_cov,
            "Avg Length": cola_avg_len,
            "Extra": serialize_extra(best_a),
        })

        # -----------------------------
        # Weighted E-value Aggregation:
        # WECA(calibrator), UR-WECA(calibrator)
        # -----------------------------
        for p_to_e in P_TO_E_LIST:
            for random_flag in [False, True]:
                e_sets, e_lengths, e_cov, e_avg_len, w_star = Evalue_aggregation_weighted(
                fitted_model_list, X_cal, y_cal, calib_sets_idx,
                X_test, y_test, alpha=alpha, U_test=U_test,
                M=M, seed=seed, P_TO_E=p_to_e,
                B=B, Random=random_flag)

                method_name = f"UR-WECA({p_to_e})" if random_flag else f"WECA({p_to_e})"

                results.append({
                    "Dataset": dataset_name,
                    "Seed": seed,
                    "Method": method_name,
                    "P_TO_E": p_to_e,
                    "Random": random_flag,
                    "Coverage": e_cov,
                    "Avg Length": e_avg_len,
                    "Extra": serialize_extra(w_star),
                })

        all_results.extend(results)

    return all_results


def main():

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for dataset_name, dataset_config in DATASETS.items():
        dataset_results = run_one_dataset(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            seeds=seeds,
            alpha=alpha,
            M=M,
            B=B,
        )

        all_results.extend(dataset_results)

    df = pd.DataFrame(all_results)

    results_path = results_dir / "all_datasets_results.csv"
    summary_path = results_dir / "all_datasets_summary.csv"

    df.to_csv(results_path, index=False)

    summary = (
        df.groupby(["Dataset", "Method"], as_index=False)
          .agg(
              Coverage_mean=("Coverage", "mean"),
              Coverage_sd=("Coverage", "std"),
              Length_mean=("Avg Length", "mean"),
              Length_sd=("Avg Length", "std"),
          )
    )

    summary.to_csv(summary_path, index=False)

    print("\nSummary:")
    print(summary)

    print(f"\nSaved full results to: {results_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()