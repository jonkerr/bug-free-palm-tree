"""
Select models and optimal hyper-parameters to be used in ensemble model.
----
Lots used from: EDA_Spike/Part 4_Model_v1.ipynb
"""

import numpy as np
import pandas as pd

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

    Parameters:
    - split_type: The type of data split to use. Options: 'std', 'date'.

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


print("\nmetrics for baseline models\n")
print(get_metrics(baseline_models))


def tune_random_forest():
    """
    Tune the hyperparameters of a Random Forest model using GridSearchCV to optimize the ROC-AUC score.
    And store the best model in the tuned_models list.

    Parameters:
    - split_type: The type of data split to use. Either 'std' for standard split or 'date' for time series split.

    Returns:
    - None
    """
    X_train, y_train = data["X_train"], data["y_train"].values.ravel()

    model = RandomForestClassifier(random_state=42)

    # hyperparameter grid
    param_grid = {
        "n_estimators": [400, 500, 600, 700],
        "max_depth": [20, 30, 40, 50],
        "min_samples_split": [2, 5, 10, 15],
        "min_samples_leaf": [1, 2, 3],
        "bootstrap": [True, False],
    }

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)

    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f"model: {model.__class__.__name__}")
    print(f"best params: {best_params}")
    print(f"best score: {best_score}\n")

    best_model = model.set_params(**best_params)
    tuned_models.append(best_model)

print("\nmetrics for tuned models\n")
tune_random_forest()
print(get_metrics(tuned_models))
