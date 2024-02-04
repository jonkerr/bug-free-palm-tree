"""
Select models and optimal hyper-parameters to be used in ensemble model.
----
Lots used from: EDA_Spike/Part 4_Model_v1.ipynb
"""

# import libraries
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.linear_model import (
    Lasso,
    LassoCV,
    Ridge,
    RidgeCV,
    ElasticNetCV,
    LogisticRegression,
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import xgboost as xgb

# ignore warnings
import warnings

# there is a convergence warning for logistic regression,
# line 152 is causing it but no solution yet
warnings.filterwarnings("ignore")

CLEAN_DATA_PATH = "./clean_data/"
TRAINING_PATH = "./training_data/"


def get_training_data(split_type="std"):
    """
    need to get files in the form:
    paths = ['X_train_std.csv', 'y_train_std.csv', 'X_test_std.csv', 'y_test_std.csv']
    """

    def format_name(fname):
        return f"{TRAINING_PATH}/{fname}_{split_type}.csv"

    files = ["X_train", "y_train", "X_test", "y_test"]
    data = {fname: pd.read_csv(format_name(fname)) for fname in files}
    return data


data = get_training_data()
# print(data['X_train'].shape)

baseline_models = [
    LogisticRegression(random_state=42),
    DecisionTreeClassifier(random_state=42),
    RandomForestClassifier(random_state=42),
]

tuned_models = []


def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    """
    Train and evaluate a model (helper function).

    Parameters:
    - model: The model to be trained and evaluated.
    - X_train: The training features.
    - y_train: The training labels.
    - X_test: The test features.
    - y_test: The test labels.

    Returns:
    - results: A dictionary containing the evaluation metrics.
    """

    # Dictionary to store metrics
    metrics = {
        "Accuracy": accuracy_score,
        "Precision": precision_score,
        "Recall": recall_score,
        "F1 Score": f1_score,
        "AUC-ROC": roc_auc_score,
    }

    results = {}

    model.fit(X_train, y_train.values.ravel())  # Train the model
    Y_pred = model.predict(X_test)  # Test the model
    Y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Evaluate the model
    for metric_name, metric_func in metrics.items():
        if metric_name == "AUC-ROC":
            score = metric_func(y_test, Y_pred_proba)
        else:
            score = metric_func(y_test, Y_pred)
        results[metric_name] = score

    return results


def get_metrics(models):
    """
    Get the evaluation metrics for each model.

    Returns:
    - df: A dataframe containing the evaluation metrics for each model.
    """

    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    results = {}

    for model in models:
        model_name = model.__class__.__name__
        metrics = train_and_evaluate(model, X_train, y_train, X_test, y_test)
        results[model_name] = metrics

    df = pd.DataFrame(results).T
    return df


# In order to generalize GridSearchCV, we need to create a function that takes in
# a model and a param_grid and returns the best model

# There needs to be a matching param_grid for each model in the baseline_models list
param_grids = {
    LogisticRegression: {
        "C": np.logspace(-4, 4, 20),
        "penalty": ["l1", "l2"],
        "solver": ["liblinear", "saga"],
        "max_iter": [
            200,
            300,
            400,
            500,
        ],  # convergence warning here but not in jupyter 🤔
        "class_weight": ["balanced", None],
    },
    DecisionTreeClassifier: {
        "criterion": ["gini", "entropy", "log_loss"],
        "max_depth": [5, 10, 20, 30, None],
        "min_samples_split": [2, 3, 4, 5, 10],
        "min_samples_leaf": [1, 2, 4, 6, 8, 10],
        "max_features": ["sqrt", "log2", None],
    },
    RandomForestClassifier: {
        "n_estimators": [500, 600, 700],
        "max_features": ["sqrt", "log2"],
        "max_depth": [20, 25, 30, 35],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False],
    },
}


def tune_model(model_instance, param_grid, scoring="roc_auc"):
    """
    Tune the hyperparameters of a model using GridSearchCV. Default scoring is optimized for ROC-AUC.

    Parameters:
    - model_instance: The model to be tuned.
    - param_grid: The hyperparameter grid to be searched.
    - scoring: The scoring metric to be optimized.

    Returns:
    - best_model: The best model with the optimized hyperparameters.
    """
    # Clone the model to avoid side-effects like changing the baseline model list
    model_instance = clone(model_instance)

    X_train, y_train = data["X_train"], data["y_train"].values.ravel()
    grid_search = GridSearchCV(
        model_instance, param_grid, cv=5, scoring=scoring, n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f"model: {model_instance.__class__.__name__}")
    print(f"best params: {best_params}")
    print(f"best score: {best_score}\n")

    best_model = model_instance.set_params(**best_params)
    return best_model


def get_tuned_models(baseline_models, param_grids, scoring="roc_auc"):
    """
    Tune the hyperparameters of the baseline models and return the best models.

    Parameters:
    - baseline_models: A list of baseline models.
    - param_grids: A dictionary containing the hyperparameter grids for each model.
    - scoring: The scoring metric to be optimized.

    Returns:
    - tuned_models: A list of the best models with the optimized hyperparameters.
    """
    tuned_models = []
    for model in baseline_models:
        model_class = model.__class__
        param_grid = param_grids[model_class]
        best_model = tune_model(model, param_grid, scoring)
        tuned_models.append(best_model)
    return tuned_models


tuned_models = get_tuned_models(baseline_models, param_grids, "roc_auc")

print("\nmetrics for baseline models\n")
print(get_metrics(baseline_models))

print("\nmetrics for tuned models\n")
print(get_metrics(tuned_models))
