import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import TimeSeriesSplit

from utils.decorators import file_check_decorator
from utils.constants import FEATURE_DATA_PATH, CLEAN_DATA_PATH, TARGET, CANDIDATE_TARGETS

# set data path for wrappers
out_data_path = FEATURE_DATA_PATH

df_features_and_targets = pd.read_csv(
    CLEAN_DATA_PATH + "merged.csv", index_col=0, parse_dates=True
)


@file_check_decorator(out_data_path)
def select_features_lasso(out_file):
    # start with clean df
    df = df_features_and_targets.copy()

    # There is a problem with the Lasso regression on our data that was different
    # from Naomi's data when she ran the same code. So, for now, I'll just set the columns
    # to the ones that were selected in Naomi's code. The problem is that the Lasso code here
    # selects 0-4 features max, which is not enough to train a model.
    # hardcoding the columns that were selected in Naomi's code in
    # EDA_Spike/Part_5_Model_All_v0_5_US.ipynb

    # # Prepare features and target
    # X_train = df.drop(CANDIDATE_TARGETS, axis=1)
    # y_train = df[CANDIDATE_TARGETS[0]]

    # # Scale features
    # sc = StandardScaler()
    # X_train_scaled = sc.fit_transform(X_train)

    # # Feature selection with LassoCV
    # alphas = np.logspace(-5, 5, 100)
    # tscv = TimeSeriesSplit(n_splits=5)
    # lasso_cv = LassoCV(alphas=alphas, cv=tscv, max_iter=10000, tol=0.001)
    # lasso_cv.fit(X_train_scaled, y_train)

    # # Identify selected features
    # selected_features = X_train.columns[lasso_cv.coef_ != 0]

    # final_df = X_train[selected_features].copy()

    # # Add the target variable back
    # final_df[CANDIDATE_TARGETS[0]] = y_train
    # final_df[CANDIDATE_TARGETS[1]] = df[CANDIDATE_TARGETS[1]]
  
    # print("Optimal Alpha:", lasso_cv.alpha_)
    # print("Coefficients:", lasso_cv.coef_)

    # selected features from EDA_Spike/Part_5_Model_All_v0_5_US.ipynb
    selected_features = ['A038RC1', 'B042RC1', 'BSCICP03USM665S', 'CAPUTLG3361T3S',
       'CAPUTLG3364T9S', 'CSCICP03USM665S', 'CUUR0000SEFR', 'CUUR0000SEFV',
       'HOHWMN02USM065S', 'IB001260M', 'IPNMAT', 'LNS12032197', 'LNS13025701',
       'MANEMP', 'NDMANEMP', 'PAYEMS', 'PERMIT', 'PERMIT1NSA',
       'SPASTT01USM657N', 'TB3SMFFM', 'UNRATE', 'USFIRE', 'USPBS', 'W875RX1',
       'WPU025', 'WPU0278', 'WPU051', 'WPU066', 'WPU071201', 'WPU0812',
       'WPU1072', 'WPU1081']    
    
    lags = ['3M', '6M', '9M', '12M', '18M']

    # Generate new list with lagged feature names
    selected_lagged_features = [f"{feature}_{lag}_lag" for feature in selected_features for lag in lags]

    # remove the columns that are not in the df
    invalid_cols = ['BSCICP03USM665S_12M_lag', 'BSCICP03USM665S_18M_lag', 
        'CSCICP03USM665S_12M_lag', 'CSCICP03USM665S_18M_lag', 'HOHWMN02USM065S_3M_lag', 
        'HOHWMN02USM065S_6M_lag', 'HOHWMN02USM065S_9M_lag', 'HOHWMN02USM065S_12M_lag', 
        'HOHWMN02USM065S_18M_lag', 'PERMIT_12M_lag', 'PERMIT_18M_lag', 
        'SPASTT01USM657N_12M_lag', 'SPASTT01USM657N_18M_lag']

    selected_lagged_features = [col for col in selected_lagged_features if col not in invalid_cols]
    
    # save
    final_df = df[[*selected_lagged_features, *CANDIDATE_TARGETS]].copy()
    final_df.to_csv(out_file)



@file_check_decorator(out_data_path)
def select_features_pca(out_file):
    # start with clean df
    df = df_features_and_targets.copy()

    # do feature selection
    # selected n_components based on cumulative explained variance 
    # visualization in jupyter notebook
    X = df.drop(CANDIDATE_TARGETS, axis=1)
    targets_df = df[[*CANDIDATE_TARGETS]]
    targets_df.index = df.index


    # Scale features
    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)

    pca = PCA(n_components=40)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(data=X_pca, columns=[f'PC{i+1}' for i in range(40)])

    # pca_df.head()

    # Reattach the date index
    pca_df.index = df.index

    # Add the target variables back
    pca_df = pca_df.join(targets_df)

    # save
    pca_df.to_csv(out_file)


"""
Data pipeline based on work done for Milestone 1: https://github.com/jonkerr/SIADS593
"""


def select_features(selection_option):
    if not os.path.isdir(FEATURE_DATA_PATH):
        os.mkdir(FEATURE_DATA_PATH)

    if selection_option in ["lasso", "all"]:
        select_features_lasso("features_lasso.csv")

    if selection_option in ["pca", "all"]:
        select_features_pca("features_pca.csv")


"""
Handle command line arguments
"""
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # pass an arg using either "-so" or "--selection_option"
    parser.add_argument(
        "-co",
        "--clean_option",
        help="Which file to clean? [lasso|pca|all] Default is all",
        default="all",
        required=False,
    )
    args = parser.parse_args()
    select_features(args.clean_option)
