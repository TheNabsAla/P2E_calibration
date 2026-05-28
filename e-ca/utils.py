
import numpy as np
import warnings
from scipy.optimize import root_scalar
import openml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_dataset(name_or_id):
    """Load dataset from OpenML and keep only numeric, non-missing rows."""
    task = openml.tasks.get_task(name_or_id)
    X_df, y = task.get_X_and_y(dataset_format="dataframe")

    # convert target to Series
    y = pd.Series(y, name="target")

    # keep only numeric feature columns
    X_df = X_df.select_dtypes(include=[np.number])

    # force target numeric; text targets become NaN
    y = pd.to_numeric(y, errors="coerce")

    # remove rows with NaN, inf, -inf in X or y
    data = X_df.copy()
    data["target"] = y

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna()

    X = data.drop(columns="target").to_numpy(dtype=float)
    y = data["target"].to_numpy(dtype=float)

    return X, y


def split_data(X, y, seed):
    # 15% test, 85% temporary
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=seed
    )

    # From the remaining 85%, take 50/85 as train and 35/85 as calibration.
    # This gives total proportions:
    # train = 0.85 * (50/85) = 0.50
    # cal   = 0.85 * (35/85) = 0.35
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_temp,
        y_temp,
        test_size=35 / 85,
        random_state=seed,
    )

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train)
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

    X_cal = scaler_X.transform(X_cal)
    y_cal = scaler_y.transform(y_cal.reshape(-1, 1)).ravel()

    X_test = scaler_X.transform(X_test)
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    return X_train, X_cal, y_train, y_cal, X_test, y_test



def random_subsample(X, Y, seed=None):
    # to generate calibration sets of different sizes
    rng = np.random.default_rng(seed)
    n = len(X)
    # 1. Determine a random fraction between 0.60 and 1.00
    fraction = rng.uniform(0.60, 1.00)
    sample_size = int(n * fraction)
    
    # 2. Randomly select indices without replacement
    indices = rng.choice(n, size=sample_size, replace=False)
    
    # 3. Return the subset slices
    return indices


def conformal_quantile(scores, alpha):
    scores = np.sort(np.asarray(scores, dtype=float))
    n = len(scores)

    rank = int(np.ceil((n + 1) * (1 - alpha)))

    if rank > n:
        return np.inf

    if rank <= 0:
        return float(scores[0])

    return float(scores[rank - 1])




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
def get_C_s(alpha, n_calib):
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
    return C, s


# Return P2E calibrator    
def f_p_to_e(x, alpha, C, s):
    f_alpha = logistic(C * (alpha - s))
    f = logistic(C * (x - s))
    return f / (alpha * f_alpha)


def get_p_to_e(P_TO_E, alpha, n_calib):
    if P_TO_E is None:
        warnings.warn("No P_TO_E calibrator provided; using 'log'.")
        P_TO_E = "log"

    if callable(P_TO_E):
        return P_TO_E

    if P_TO_E == "log":
        return lambda p: -np.log(np.clip(p, 1e-12, 1.0))

    if P_TO_E == "linear":
        return lambda p: 2 * (1 - p)

    if P_TO_E == "sqrt":
        return lambda p: 1 / np.sqrt(np.clip(p, 1e-12, 1.0)) - 1
    
    if P_TO_E == "P2E":
        C, s = get_C_s(alpha, n_calib)
        return lambda p: f_p_to_e(p, alpha, C, s)
    
    if P_TO_E == "AoN":
        return lambda p: (1.0 / alpha) * (np.asarray(p) <= alpha)

    raise ValueError(f"Unknown P_TO_E calibrator: {P_TO_E}")


def grid_set_length(cp_set, y_grid):
    """
    Length of a CP set represented by selected grid points.
    Works for disjoint intervals.

    cp_set : array of selected grid points, e.g. y_grid[mask]
    y_grid : full grid used to build cp_set
    """
    if len(cp_set) == 0:
        return 0.0

    step = y_grid[1] - y_grid[0]
    return len(cp_set) * step