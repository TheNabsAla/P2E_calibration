import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def _ensure_exists(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File not found: {path}\n Make sure the dataset is placed under this path."
        )

def load_boston():
    path = "datasets/Boston.csv"
    _ensure_exists(path)
    df = pd.read_csv(path)
    y = df["medv"].to_numpy()
    X = df.drop(columns=["medv"]).to_numpy()
    cfg = dict(name="Boston", n_train=400, n_test=None, K=5, alpha=0.10, lambda_=0.01, ntree=200)
    return X, y, cfg

def load_parkinson():
    path = "datasets/merged_dataset.csv"
    _ensure_exists(path)
    df = pd.read_csv(path)
    y = df["total_UPDRS"].to_numpy()

    drop_cols = [0, 1, 5, 6, 7, 10, 11, 12]
    Xdf = df.drop(df.columns[drop_cols], axis=1).copy()

    if Xdf.shape[1] >= 12:
        scaler = StandardScaler()
        Xdf.iloc[:, 0:12] = scaler.fit_transform(Xdf.iloc[:, 0:12])

    X = Xdf.to_numpy()
    cfg = dict(name="Parkinsons_UPDRS", n_train=3000, n_test=None, K=5, alpha=0.10, lambda_=0.01, ntree=200)
    return X, y, cfg

def load_crime():
    path = "datasets/crimedata.csv"
    _ensure_exists(path)
    df = pd.read_csv(path, sep=",", header=0, na_values="?")
    df = df.loc[:, df.isna().sum() == 0].copy()
    if df.shape[1] > 3:
        df = df.drop(df.columns[0:3], axis=1)

    y = df["ViolentCrimesPerPop"].to_numpy()
    X = df.iloc[:, 0:min(99, df.shape[1]-1)].to_numpy()
    cfg = dict(name="CommunitiesCrime", n_train=1000, n_test=None, K=10, alpha=0.10, lambda_=0.01, ntree=200)
    return X, y, cfg

def load_news():
    path = "datasets/news_pop.csv"
    _ensure_exists(path)
    df = pd.read_csv(path, sep=",", header=0, na_values="?")
    df.columns = df.columns.str.strip()

    y = np.log(df["shares"].to_numpy())
    Xdf = df.drop(columns=["shares"]).copy()

    drop_idx = [3, 4, 18]
    drop_idx = [i for i in drop_idx if i < Xdf.shape[1]]
    if drop_idx:
        Xdf = Xdf.drop(Xdf.columns[drop_idx], axis=1)

    X = Xdf.to_numpy()
    cfg = dict(name="NewsPopularity", n_train=10000, n_test=2500, K=10, alpha=0.10, lambda_=0.20, ntree=250)
    return X, y, cfg

def load_abalone():
    path = "datasets/abalone.csv"
    _ensure_exists(path)
    df = pd.read_csv(path)

    y = df["Rings"].to_numpy()
    X1 = pd.get_dummies(df["Sex"], drop_first=False).to_numpy()
    X2 = df.iloc[:, 1:8].to_numpy()
    X = np.hstack([X1, X2])

    cfg = dict(name="Abalone", n_train=4000, n_test=None, K=10, alpha=0.10, lambda_=0.01, ntree=200)
    return X, y, cfg

_LOADERS = {
    "boston": load_boston,
    "abalone": load_abalone,
    "crime": load_crime,
    "news": load_news,
    "parkinson": load_parkinson,
    "parkinsons": load_parkinson,
    "updrs": load_parkinson,
}

def load_dataset(name: str):
    """
    Return (X, y, cfg) for a given dataset name (case-insensitive).
    cfg includes: name, n_train, n_test (None = auto n - n_train), K, alpha, lambda_, ntree.
    """
    key = str(name).strip().lower()
    if key not in _LOADERS:
        opts = ", ".join(sorted(_LOADERS.keys()))
        raise ValueError(f"Unknown dataset '{name}'. Available: {opts}")
    X, y, cfg = _LOADERS[key]()
    if not isinstance(X, np.ndarray): X = np.asarray(X)
    if not isinstance(y, np.ndarray): y = np.asarray(y).ravel()
    if y.ndim != 1: y = y.ravel()
    assert X.shape[0] == y.shape[0], f"Mismatch: X has {X.shape[0]} rows, y has {y.shape[0]}"
    assert cfg["n_train"] < y.shape[0], "n_train must be less than total n"
    if cfg["n_test"] is not None:
        assert cfg["n_train"] + cfg["n_test"] <= y.shape[0], "n_train + n_test exceeds dataset size"
    return X, y, cfg
