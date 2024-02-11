import pandas as pd
import os

from utils.decorators import file_check_decorator
from utils.constants import FEATURE_DATA_PATH, CLEAN_DATA_PATH

# set data path for wrappers
out_data_path = FEATURE_DATA_PATH

df_features_and_targets = pd.read_csv(CLEAN_DATA_PATH + 'merged.csv', index_col=0, parse_dates=True)

@file_check_decorator(out_data_path)
def select_features_lasso(out_file):
    # start with clean df
    df = df_features_and_targets.copy()

    # do feature selection

    # save    
    df.to_csv(out_file)    
   

@file_check_decorator(out_data_path)
def select_features_pca(out_file):
    # start with clean df
    df = df_features_and_targets.copy()

    # do feature selection

    # save    
    df.to_csv(out_file)        
    
    
'''
Data pipeline based on work done for Milestone 1: https://github.com/jonkerr/SIADS593
'''
def select_features(selection_option):
    if not os.path.isdir(FEATURE_DATA_PATH):
        os.mkdir(FEATURE_DATA_PATH)
    
    if selection_option in ['lasso', 'all']:
        select_features_lasso('features_lasso.csv')

    if selection_option in ['pca', 'all']:
        select_features_pca('features_pca.csv')


'''
Handle command line arguments
'''
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # pass an arg using either "-so" or "--selection_option"
    parser.add_argument('-co', '--clean_option',
                        help='Which file to clean? [lasso|pca|all] Default is all',
                        default="all",
                        required=False)
    args = parser.parse_args()
    select_features(args.clean_option)
