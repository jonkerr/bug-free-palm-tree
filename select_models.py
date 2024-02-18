"""
Select models and optimal hyper-parameters to be used in ensemble model.
----
Lots used from: EDA_Spike/Part 4_Model_v1.ipynb
"""

# import libraries
import time
from functools import wraps
import json
import os

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
    BaggingClassifier,
)
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import xgboost as xgb
from utils.constants import (
    SEED, 
    TRAINING_DATA_PATH, 
    SPLIT_DATA_PATH,
    SCORING,
    SPLIT_TYPE,
    FEATURE_TYPE,
    TARGET
)

def get_training_data(split_type=SPLIT_TYPE, feature_type=FEATURE_TYPE, target=TARGET):
    """
    need to get files in the form:
    paths = ['X_train_std.csv', 'y_train_std.csv', 'X_test_std.csv', 'y_test_std.csv']
    """

    def format_name(fname):
        
        name = fname
        if feature_type is not None and not 'y_' in name:
            name += "_" + feature_type
            
        return f"{SPLIT_DATA_PATH}{name}_{target}_{split_type}.csv"
                
        #if feature_type:
        #    return f"{TRAINING_DATA_PATH}{fname}_{feature_type}_{split_type}.csv"
        #return f"{TRAINING_DATA_PATH}{fname}_{split_type}.csv"

    files = ["X_train", "y_train", "X_test", "y_test"]
    data = {fname: pd.read_csv(format_name(fname)) for fname in files}
    if feature_type:
        print(f"**Data is using the '{feature_type}' feature type**")
    print(f"**Data is using the '{split_type}' split type**\n")
    return data


#data = get_training_data()

baseline_models = [
    LogisticRegression(random_state=SEED),
    KNeighborsClassifier(),  # no random_state
    SVC(
        probability=True, random_state=SEED
    ),  # probability=True needed for roc_auc_score
    GaussianNB(),
    BernoulliNB(),
    GaussianProcessClassifier(random_state=SEED),
    DecisionTreeClassifier(random_state=SEED),
    RandomForestClassifier(random_state=SEED),
    GradientBoostingClassifier(random_state=SEED),
    AdaBoostClassifier(random_state=SEED),
    ExtraTreesClassifier(random_state=SEED),
    BaggingClassifier(random_state=SEED),
    xgb.XGBClassifier(random_state=SEED),
]

model_needs_scaling = [
    "LogisticRegression",
    "KNeighborsClassifier",
    "SVC",
    "GaussianProcessClassifier",  # might benefit from scaling (it does)
]

ignore_in_phase_2 = [
    'DecisionTreeClassifier',
    'AdaBoostClassifier',
]

# There needs to be a matching param_grid for each model in
# the baseline_models list for the tuning process to work
param_grids = {
    LogisticRegression: {
        "C": np.logspace(-3, 0, 50),  # 50 values between 10^-3 and 10^0
        "solver": ["liblinear", "lbfgs"],
        "tol": [1e-4, 1e-3, 0.01, 0.1, 1],
        "max_iter": [
            100,
            200,
            500,
            1000,
            1500,
        ],
        "class_weight": ["balanced", None],
    },
    KNeighborsClassifier: {
        "n_neighbors": [5, 10, 15, 20, 25, 30],
        "weights": ["uniform", "distance"],
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        "p": [1, 2],
        "metric": ["minkowski", "euclidean", "manhattan", "chebyshev"],
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
        "max_features": ["sqrt"],  # reuse the solution of the previous calls.
        "max_depth": [20, 25, 30, 35],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [False],  # reuse the solution of the previous calls.
    },
    SVC: {
        "C": np.linspace(1, 4, 20),  # 20 values between 1 and 4
        "gamma": ["scale", "auto"],
        "kernel": ["rbf", "poly", "sigmoid", "linear"],
        "class_weight": ["balanced", None],
    },
    GradientBoostingClassifier: {  # tuning too many parameters makes this too slow
        "n_estimators": [150, 200, 300],
        "loss": ["exponential"],  # reuse the solution of the previous calls.
        "learning_rate": [0.4, 0.35, 0.3, 0.25],
        "max_depth": [5, 6, 7, 8, 9],
        "warm_start": [True],  # reuse the solution of the previous calls.
    },
    AdaBoostClassifier: {
        "n_estimators": [1, 2, 3, 4, 5, 10, 15, 25],
        "learning_rate": [20, 11, 10, 5, 1, 0.5, 0.1, 0.01],
        "estimator": [
            None,
            RandomForestClassifier(),
        ],
    },
    GaussianNB: {
        "var_smoothing": np.logspace(0, -9, 100),  # 100 values between 10^0 and 10^-9
    },
    BernoulliNB: {
        "alpha": np.linspace(1, 10, 50),  # 50 values between 1 and 10
        "binarize": np.linspace(0, 1, 20),  # 20 values between 0 and 1
        "fit_prior": [True, False],
    },
    GaussianProcessClassifier: {
        "max_iter_predict": [20, 50, 100, 200],
        "n_restarts_optimizer": [0, 1, 2, 3, 4],
        "warm_start": [True, False],
    },
    ExtraTreesClassifier: {
        "n_estimators": [100, 200, 300, 400, 500],
        "criterion": ["gini", "entropy", "log_loss"],
        "min_samples_split": [2, 3, 4, 5, 10],
        "max_features": ["sqrt", "log2", None],
    },
    BaggingClassifier: {
        "estimator": [None, RandomForestClassifier()],
        "n_estimators": [5, 10, 20],
        "bootstrap": [True, False],
        "bootstrap_features": [True, False],
    },
    xgb.XGBClassifier: {
        "booster": ["gbtree", "dart"],
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 15, 20],
        "learning_rate": [0.1, 0.2, 0.3],
    },
}


def timing_decorator(func):
    """
    A decorator to measure the execution time of a function.

    Parameters:
    - func: The function to be measured.

    Returns:
    - wrapper: The wrapper function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(
            f"Execution time of {func.__name__} function: {round((end_time - start_time), 2)} seconds"
        )
        return result

    return wrapper


def scale_data(X_train, X_test=None):
    """
    Use StandardScaler to scale the data for models that require it.

    Parameters:
    - X_train: The training features.
    - X_test: The test features. Optional if you only want to scale the training data.

    Returns:
    - X_train_scaled: The scaled training features.
    - X_test_scaled: The scaled test features.
    """
    # Make a copy of the data to avoid side-effects like changing the original data
    X_train = X_train.copy()
    X_test = X_test.copy() if X_test is not None else None

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = None

    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled


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
        "recall": recall_score,
        "roc_auc": roc_auc_score,
        "accuracy": accuracy_score,
        "precision": precision_score,
        "f1": f1_score,
    }

    results = {}

    # Clone the model to avoid side-effects like overfitting
    model = clone(model)

    model.fit(X_train, y_train.values.ravel())  # Train the model
    Y_pred = model.predict(X_test)  # Test the model
    Y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Evaluate the model
    for metric_name, metric_func in metrics.items():
        if metric_name == "roc_auc":
            score = metric_func(y_test, Y_pred_proba)
        else:
            score = metric_func(y_test, Y_pred)
        results[metric_name] = round(score, 3)

    return results

    

def get_metrics(
    models, data, X_train=None, X_test=None, scoring=SCORING
):
    """
    Get the evaluation metrics for each model.

    Parameters:
    - models: A list of model instances.
    - X_train: The training features. Default is X_train from the base training set
    - X_test: The test features. Default is X_test from the base training set
    - scoring: The scoring metric to be used. Default is 'roc_auc'.

    Returns:
    - df: A dataframe containing the evaluation metrics for each model.
    """
    
    if X_train is None:
        X_train = data["X_train"]
    if X_test is None:        
        X_test = data["X_test"]
        
    y_train, y_test = data["y_train"], data["y_test"]

    results = {}

    for model in models:
        model_name = model.__class__.__name__
        # create local copies for potential scaling
        X_train_temp, X_test_temp = X_train.copy(), X_test.copy()

        if model_name in model_needs_scaling:
            X_train_temp, X_test_temp = scale_data(X_train_temp, X_test_temp)

        # use temp copies for training and evaluation
        metrics = train_and_evaluate(model, X_train_temp, y_train, X_test_temp, y_test)
        results[model_name] = metrics

    df = pd.DataFrame(results).T
    df = df.sort_values(by='recall', ascending=False)
    return df


@timing_decorator
def tune_model(model, param_grid, data, X_train=None, scoring=SCORING):
    """
    Tune the hyperparameters of a model using GridSearchCV. Default scoring is optimized for ROC-AUC.

    Parameters:
    - model: The model to be tuned.
    - param_grid: The hyperparameter grid for the model.
    - X_train: The training features. Default is X_train from the base training set
    - scoring: The scoring metric to be used. Default is 'roc_auc'.

    Returns:
    - best_model: The best model with the optimized hyperparameters.
    - best_params: Used to serialize/rehydrate previously optimized models.
    """
    # Clone the model to avoid side-effects like changing the baseline model list
    model = clone(model)
    model_name = model.__class__.__name__
    
    if X_train is None:
        X_train = data["X_train"]
    y_train = data["y_train"].values.ravel()

    if model_name in model_needs_scaling:
        X_train, _ = scale_data(X_train)

    print(f"Tuning hyperparameters for {model_name}...")
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scoring, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f"model: {model_name}")
    print(f"best params: {best_params}")
    print(f"best {scoring} score: {round(best_score, 3)}")

    best_model = model.set_params(**best_params)
    return best_model, best_params


def rehydrate_models(json_file):    
    """
    Tune the hyperparameters of a model using GridSearchCV. Default scoring is optimized for ROC-AUC.

    Parameters:
    - json_file: Name of the json file to read from

    Returns:
    - tuned_models: The optimally tuned models
    - tuned_model_names: The names of the tuned models
    - best_model_params: The params used for the optimal models
    
    """
    if not os.path.exists(json_file):
        return [],[]
    
    best_model_params = {}
    tuned_models = []
    tuned_model_names = []

    with open(json_file, 'r') as file:
        # Reading from json file
        best_model_params = json.load(file)
            
    for model in baseline_models:
        model_name = model.__class__.__name__      
        if model_name in best_model_params.keys():
            best_params = best_model_params[model_name]
            best_model = model.set_params(**best_params)
            tuned_models.append(best_model)   
            tuned_model_names.append(model_name)
            
    return tuned_models, tuned_model_names, best_model_params
            
import pickle    

def get_tuned_models(param_grids, data, X_train=None, scoring=SCORING, stage=1, rehydrate=False):
    """
    Tune the hyperparameters of the baseline models and return the best models.

    Parameters:
    - param_grids: A dictionary containing the hyperparameter grids for each model.
    - X_train: The training features. Default is X_train from the base training set
    - scoring: The scoring metric to be used. Default is 'roc_auc'.

    Returns:
    - tuned_models: A list of the best models with the optimized hyperparameters.
    """
    #json_file = f'stage{stage}_params.json'
    #tuned_models_old, tuned_model_names, best_model_params = rehydrate_models(json_file) if rehydrate else ([],[], {})
    tuned_models = {}
               
    if X_train is None:
        X_train=data["X_train"]
    
    for model in baseline_models:
        model_class = model.__class__
        model_name = model_class.__name__

        # rehydrate if file exists and permitted to do so
        pkl_name = get_pickle_name(model_name, stage, feature, target, split_type)
        if rehydrate and os.path.exists(pkl_name):
            with open(pkl_name, "rb") as file:
                tuned_models[pkl_name] = pickle.load(file)
            continue
                
        param_grid = param_grids[model_class]
        best_model, best_params = tune_model(model, param_grid, data, X_train, scoring=scoring)
        
        tuned_models[pkl_name] = best_model
        #tuned_models.append(best_model)
        print(f"{len(tuned_models.keys())}/{len(baseline_models)} models tuned\n")
        # store best params for rehydration
        #best_model_params[model_name] = best_params
        
    # save best params
    for path, model in tuned_models.items():
        with open(path, "wb") as file:
            pickle.dump(model, file)    
    
    """
    with open(json_file, "w") as file:
        # Some algos wrap others, which makes them non-serializable
        # Remove from list and we'll need to CV them each time
        del best_model_params['AdaBoostClassifier']
        del best_model_params['BaggingClassifier']
        # save to JSON
        json.dump(best_model_params, file)
    """

    return tuned_models.values()


def get_fully_qualified_path(stage, feature, target, split_type):
    folder = f'./model_data/stage_{stage}/{target}_{split_type}'
    if feature is not None:
        folder += f'_{feature}'
    folder += '/'
    return folder


def get_simple_path(stage):
    return f'./model_data/stage_{stage}/'


def get_pickle_name(model_name, stage, feature, target, split_type, use_fully_qualified_path=True):
    '''
    The fully qualified path is interesting to save optimal model parameters for each stage/feature selection option/target/split type/etc.
    However, there is a concern that this could lead to overfitting.  Use the following test:
    1. Train all paramaters on optimal, per config setting
    2. Record outputs and find "top model"
    3. Run again with a different seed value.  The hyperparameters won't change since they've alread been saved but we can find out how sensitive the results are base on seed.
    4. If results are significantly different, then use the simple path to use the same settings for all configurations.
    '''
    folder = get_fully_qualified_path(stage, feature, target, split_type) if use_fully_qualified_path else get_simple_path(stage)
    
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    return folder + model_name + '.pkl'

def get_probs(models, data):
    """
    Collect probability predictions from each model for the positive class.
    This is used as the input for the stage 2 models where we are trying to
    select the best models to use in the ensemble.

    Parameters:
    - models: A list of model instances.

    Returns:
    - train_probs: A DataFrame containing the training set probability predictions.
    - test_probs: A DataFrame containing the test set probability predictions.
    """
    train_probs = pd.DataFrame()
    test_probs = pd.DataFrame()

    models = [
        clone(model) for model in models
    ]  # Clone the models to avoid side-effects

    for model in models:
        # Get the model's class name to use as a column name
        model_name = model.__class__.__name__
        # skip poorly performing stage 2 models
        if model_name in ignore_in_phase_2:
            continue
        # Check if the model requires scaling
        if model_name in model_needs_scaling:
            # Scale the data and use the original data for each model to avoid repeated scaling
            X_train_scaled, X_test_scaled = scale_data(data["X_train"], data["X_test"])
            # Fit the model on the scaled training data
            model.fit(X_train_scaled, data["y_train"].values.ravel())
            # Use the scaled training data for generating predictions
            train_probs[model_name] = model.predict_proba(X_train_scaled)[:, 1]
            test_probs[model_name] = model.predict_proba(X_test_scaled)[:, 1]
        else:
            # Fit the model on the original training data
            model.fit(data["X_train"], data["y_train"].values.ravel())
            # Use the original training data for generating predictions
            train_probs[model_name] = model.predict_proba(data["X_train"])[:, 1]
            test_probs[model_name] = model.predict_proba(data["X_test"])[:, 1]

    return train_probs.round(4), test_probs.round(4)


@timing_decorator
def run_comparison(data, include_baseline=True, include_stage_2=True, rehydrate=False, save_metrics=False):

    # if we are exporting the tuned models for prediction, consider making a dictionary
    print("Stage 1: Tuning models on training data...\n")
    s1_tuned_models = get_tuned_models(param_grids, data, rehydrate=rehydrate)

    if include_baseline:
        print("Stage 1: Training and evaluating baseline_models on training data...")
        s1_base_metrics = get_metrics(baseline_models, data)
        print(s1_base_metrics)
        if save_metrics:
            s1_base_metrics.to_csv("s1_baseline_metrics.csv")

    print("\nStage 1: Training and evaluating s1_tuned_models on training data...")
    s1_tuned_metrics = get_metrics(s1_tuned_models, data)
    print(s1_tuned_metrics)
    if save_metrics:
        s1_tuned_metrics.to_csv("s1_tuned_metrics.csv")

    if include_stage_2:
        # get the probability predictions for each model
        train_probs, test_probs = get_probs(s1_tuned_models, data)

        # train_probs.to_csv("train_probs.csv", index=False)
        # test_probs.to_csv("test_probs.csv", index=False)

        # train_probs = pd.read_csv("train_probs.csv")
        # test_probs = pd.read_csv("test_probs.csv")

        # get tuned models for stage 2
        print("Stage 2: Tuning models on probability data...\n")
        s2_tuned_models = get_tuned_models(param_grids, data, X_train=train_probs, stage=2)

        if include_baseline:
            print("Stage 2: Training and evaluating baseline_models on probability data...")
            s2_base_metrics = get_metrics(baseline_models, data, X_train=train_probs, X_test=test_probs)
            print(s2_base_metrics)
            if save_metrics:
                s2_base_metrics.to_csv("s2_baseline_metrics.csv")

        print("\nStage 2: Training and evaluating s2_tuned_models on probability data...")
        s2_tuned_metrics = get_metrics(s2_tuned_models, data, X_train=train_probs, X_test=test_probs)
        print(s2_tuned_metrics)
        if save_metrics:
            s2_tuned_metrics.to_csv("s2_tuned_metrics.csv")

# full run
#run_comparison()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # pass an arg using either "-so" or "--selection_option"
    parser.add_argument(
        "-rt",
        "--run_type",
        help="While run type?  Stage 1 and 2 use preexisting model params.  Full regenerates.  Targets: [stg1|stg2|full] Default is stg1",
        default="stg1",
        required=False,
    )
    parser.add_argument(
        "-ib",
        "--include_baseline",
        help="Include baseline?  [y|n] Default is n",
        default="n",
        required=False,
    )
    
    parser.add_argument(
        '-sty', '--split_type',
        help='Which split type? [date|std] Default is std',
        default="std",
        required=False
    )
    
    parser.add_argument(
        "-fo",
        "--feature_option",
        help="Which file to clean? [lasso|pca|none|all] Default is none",
        default="none",
        required=False,
    )
    
    parser.add_argument(
        '-star',
        '--split_target',
        help='Which split target? [bear|rec|all] Default is bear',
        default="bear",
        required=False
    )

    # default selection params
    include_baseline=False
    include_stage_2=False
    rehydrate=True
    save_metrics=False
    
    # default splits
    split_type = 'std'
    target=TARGET
    
    # Parse and process args
    args = parser.parse_args()
    
    # run type
    if args.run_type == 'full':
        rehydrate = False
        include_stage_2 = True
    if args.run_type == 'stg2':
        include_stage_2 = True
        
    # baseline
    if args.include_baseline == 'y':
        include_baseline = True

    # split type
    split_type = args.split_type

    # feature selection approach    
    feature_options = []
    if args.feature_option in ['lasso','all']:
        feature_options.append('lasso')
    if args.feature_option in ['pca','all']:
        feature_options.append('pca')
    if args.feature_option in ['none','all']:
        feature_options.append(None)
    

    # target
    split_targets = [] 
    if args.split_target in ['bear','all']:
        split_targets.append('bear')
    if args.split_target in ['rec','all']:
        split_targets.append('Regime')


    # put it all together
    for feature in feature_options:
        for target in split_targets:    
            data = get_training_data(split_type=split_type, feature_type=feature, target=target)  
            run_comparison(data, include_baseline=include_baseline, include_stage_2=include_stage_2, rehydrate=rehydrate, save_metrics=save_metrics)
