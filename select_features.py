import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import TimeSeriesSplit

from utils.decorators import file_check_decorator
from utils.constants import FEATURE_DATA_PATH, CLEAN_DATA_PATH, TARGET, CANDIDATE_TARGETS, SPLIT_DATA_PATH

import warnings

# set data path for wrappers
out_data_path = FEATURE_DATA_PATH

"""
df_features_and_targets = pd.read_csv(
    CLEAN_DATA_PATH + "merged.csv", index_col=0, parse_dates=True
)
"""


#@file_check_decorator(out_data_path)
def select_features_lasso(out_file, split_data, target):
    
    X_train, y_train, X_test, y_test = split_data
    
    # Scale features
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    
    # Create a Lasso model
    selected_alpha = 0.025 if target == 'bear' else .009    
    lasso_model = Lasso(alpha = selected_alpha)

    # Fit the Lasso model on the training data
    lasso_model.fit(X_train_scaled, y_train)

    # Get the selected features
    selected_features_lasso = X_train.columns[lasso_model.coef_ != 0]
    #print('selected features: ', selected_features_lasso)
    
    X_train_lasso = X_train[selected_features_lasso]
    X_test_lasso = X_test[selected_features_lasso]
    
    print(f'X_train ({target}) dimensions: ', X_train_lasso.shape)
    print(f'X_test ({target}) dimensions: ', X_test_lasso.shape)

    # save
    X_train_lasso.to_csv(SPLIT_DATA_PATH + "X_train_" + out_file)
    X_test_lasso.to_csv(SPLIT_DATA_PATH + "X_test_" + out_file)


#@file_check_decorator(out_data_path)
def select_features_pca(out_file, split_data):
       
    X_train, y_train, X_test, y_test = split_data
    
    # start with clean df
    # df = df_features_and_targets.copy()

    # do feature selection
    # selected n_components based on cumulative explained variance
    # visualization in jupyter notebook
    """
    Not needed?
    
    X = df.drop(CANDIDATE_TARGETS, axis=1)
    targets_df = df[[*CANDIDATE_TARGETS]]
    targets_df.index = df.index
    """

    # Scale features
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)

    pca = PCA(n_components=170)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    pca_train_df = pd.DataFrame(data=X_train_pca, columns=[f'PC{i+1}' for i in range(170)])
    pca_test_df = pd.DataFrame(data=X_test_pca, columns=[f'PC{i+1}' for i in range(170)])

    # pca_df.head()

    # Reattach the date index
    pca_train_df.index = X_train.index
    pca_test_df.index = X_test.index

    # Add the target variables back
    ## JK: why???
    #pca_df = pca_df.join(targets_df)

    # save
    pca_train_df.to_csv(SPLIT_DATA_PATH + "X_train_" + out_file)
    pca_test_df.to_csv(SPLIT_DATA_PATH + "X_test_" + out_file)


"""
Data pipeline based on work done for Milestone 1: https://github.com/jonkerr/SIADS593
"""
def get_split_data(target, stype):
    '''
    Get X_train, y_train, X_test, y_test associated with a particular split type and target type

    Parameters:
    - target: One of: [bear, Regime, correction]
    - stype: One of: [std, date]

    Returns:
    - X_train, y_train, X_test, y_test associated with a particular split type and target type
    '''
    prefixes = ['X_train', 'y_train', 'X_test', 'y_test']
    names = [f'{p}_{target}_{stype}.csv' for p in prefixes]
    return [pd.read_csv(SPLIT_DATA_PATH+fname, index_col=0, parse_dates=True).reset_index(names='Date') for fname in names]
    

def select_features(selection_option, split_targets, split_types):
#    if not os.path.isdir(FEATURE_DATA_PATH):
#        os.mkdir(FEATURE_DATA_PATH)

    for target in split_targets:
        for stype in split_types:
            split_data = None
            with warnings.catch_warnings(action="ignore"):
                split_data = get_split_data(target, stype) 
                       
            if selection_option in ["lasso", "all"]:
                select_features_lasso(f"lasso_{target}_{stype}.csv", split_data, target)

            if selection_option in ["pca", "all"]:
                select_features_pca(f"pca_{target}_{stype}.csv", split_data)


"""
Handle command line arguments
"""
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # pass an arg using either "-so" or "--selection_option"
    parser.add_argument(
        "-fo",
        "--feature_option",
        help="Which file to clean? [lasso|pca|all] Default is all",
        default="all",
        required=False,
    )
    
    parser.add_argument(
        '-star',
        '--split_target',
        help='Which split target? [bear|rec|all] Default is all',
        default="all",
        required=False
    )
    
    parser.add_argument(
        '-sty', '--split_type',
        help='Which split type? [date|std|all] Default is std',
        default="std",
        required=False
    )
    
    args = parser.parse_args()
    
    split_targets = [] 
    if args.split_target == 'all':
        split_targets = ['bear','Regime']
    elif args.split_target == 'bear':
        split_targets = ['bear']
    elif args.split_target == 'rec':
        split_targets = ['Regime']
        
    split_types = []
    if args.split_type == 'all':
        split_types = ['date','std']
    elif args.split_type == 'std':
        split_types = ['std']
    elif args.split_target == 'date':
        split_types = ['date']
    
    
    select_features(args.feature_option, split_targets, split_types)
