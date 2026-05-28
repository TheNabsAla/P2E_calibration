from sklearn.linear_model import (
    LinearRegression,
    Lasso,
    BayesianRidge,
    SGDRegressor
)

from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor
)

from sklearn.neural_network import MLPRegressor

DATASETS = {
    "dataset_361237": 361237,
    "dataset_361235": 361235,
    "dataset_361244": 361244,
    "dataset_361234": 361234,
   
}
# Configuration
alpha = 0.05
data_limit = 7000 
seeds = [
    42, 0, 1, 7, 10,
    13, 17, 19, 23, 29,
    31, 37, 41, 43, 47,
    53, 59, 61, 67, 71
] 


# Models
def make_models(seed):
    models = {
        "Linear": LinearRegression(),

        "Lasso": Lasso(
            alpha=0.1,
            random_state=seed,
        ),

        "RandomForest": RandomForestRegressor(
            n_estimators=50,
            random_state=seed,
            min_samples_leaf=10,
        ),

        "MLP": MLPRegressor(
            hidden_layer_sizes=(10, 5),
            learning_rate="adaptive",
            learning_rate_init=0.01,
            max_iter=1000,
            random_state=seed,
        ),

        "Bayesian": BayesianRidge(),

        "boosttree": HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=200,
            max_depth=2,
            min_samples_leaf=30,
            l2_regularization=1.0,
            random_state=seed,
        ),

        "SGDRegressor": SGDRegressor(
            max_iter=200,
            tol=1e-3,
            random_state=seed,
        ),
    }

    return models


# Method parameters
M = 512  # Grid points for aggregation methods
G = 10   # Grid for COLA_S
B = 500  # Number of weight samples for weighted methods