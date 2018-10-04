from sklearn.preprocessing import StandardScaler  # , RobustScaler
from sklearn.pipeline import Pipeline
import numpy as np

# "lr", "sgd", "rf" "lightGBM"
class ClfDef:
    """
    define pipelines and their hyperparams
    """

    def __init__(self, clf_type):

        self.trained_base_name = clf_type

        if clf_type is "lr":
            self.family = "linear"
            from sklearn.linear_model import LogisticRegression

            steps = [("scaler", StandardScaler()), ("clf", LogisticRegression())]
            self.pipeline = Pipeline(steps)
            self.param_dist = {
                "clf__intercept_scaling": [
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.6,
                    0.7,
                    0.8,
                    0.9,
                    1,
                ],
                "clf__fit_intercept": [True, False],
                "clf__C": [
                    0.000001,
                    0.00001,
                    0.0001,
                    0.001,
                    0.01,
                    0.1,
                    1.0,
                    10.,
                    100.,
                    1000.,
                ],
                "clf__penalty": ["l1", "l2"],
                "clf__max_iter": [10, 20, 30, 50],
                "clf__class_weight": [
                    "balanced",
                    None,
                    {0: 1, 1: 1},
                    {0: 1, 1: 5},
                    {0: 1, 1: 10},
                    {0: 1, 1: 50},
                    {0: 1, 1: 100},
                    {1: 1, 0: 5},
                    {1: 1, 0: 10},
                    {1: 1, 0: 50},
                    {1: 1, 0: 100},
                ],
                "clf__random_state": [0],
            }
        elif clf_type is "sgd":
            self.family = "linear"
            from sklearn.linear_model import SGDClassifier

            steps = [("scaler", StandardScaler()), ("clf", SGDClassifier())]
            self.pipeline = Pipeline(steps)
            self.param_dist = {
                "clf__loss": [
                    "hinge",
                    "log",
                    "modified_huber",
                    "squared_hinge",
                    "perceptron",
                ],
                "clf__penalty": ["l2", "l1", "elasticnet"],
                "clf__l1_ratio": [0.01, 0.05, 0.1, 0.15, 0.2, 0.35, 0.5, 0.75, 0.9, 1.],
                "clf__fit_intercept": [True, False],
                # "clf__C": [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10., 100., 1000.],
                "clf__max_iter": [10, 20, 30, 50],
                "clf__class_weight": [
                    "balanced",
                    None,
                    {0: 1, 1: 1},
                    {0: 1, 1: 5},
                    {0: 1, 1: 10},
                    {0: 1, 1: 50},
                    {0: 1, 1: 100},
                    {1: 1, 0: 5},
                    {1: 1, 0: 10},
                    {1: 1, 0: 50},
                    {1: 1, 0: 100},
                ],
                "clf__random_state": [0],
            }
        elif clf_type is "rf":
            self.family = "tree"
            from sklearn.ensemble import RandomForestClassifier

            steps = [("scaler", StandardScaler()), ("clf", RandomForestClassifier())]
            self.pipeline = Pipeline(steps)
            self.param_dist = {
                "clf__n_estimators": [10, 25, 50, 100, 250, 500],
                "clf__criterion": ["gini", "entropy"],
                "clf__min_samples_split": [2, 3, 4, 5, 6],
                "clf__min_samples_leaf": [1, 2, 3, 4, 5],
                "clf__class_weight": [
                    "balanced",
                    None,
                    {0: 1, 1: 1},
                    {0: 1, 1: 5},
                    {0: 1, 1: 10},
                    {0: 1, 1: 50},
                    {0: 1, 1: 100},
                    {1: 1, 0: 5},
                    {1: 1, 0: 10},
                    {1: 1, 0: 50},
                    {1: 1, 0: 100},
                ],
                "clf__random_state": [0],
            }
        elif clf_type is "lightGBM":
            self.family = "tree"
            import lightgbm as lgbm

            steps = [("scaler", StandardScaler()), ("clf", lgbm.LGBMClassifier())]
            self.pipeline = Pipeline(steps)
            self.param_dist = {
                "clf__n_estimators": [2, 4, 6, 8, 10, 100, 500, 1000],  # 0,
                "clf__learning_rate": [0.1, 0.5, 1],
                "clf__num_leaves": [64, 128, 256, 512, 1024],
                "clf__max_bin": [4, 8, 16, 32, 64],
                "clf__subsample": np.arange(0.5, 1.0, 0.1),
                "clf__subsample_freq": [1, 2, 4, 8, 16, 32, 64, 128, 256],
                "clf__reg_lambda": [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 0],
                "clf__eval_metric": ["binary_logloss"],
                "clf__is_unbalance": [True],
                # "clf__scale_pos_weight": []
            }
        else:
            raise NotImplementedError
