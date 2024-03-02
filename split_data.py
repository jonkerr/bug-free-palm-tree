import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# oversampling to fix the majority class
from imblearn.over_sampling import SMOTE

# project constants
from utils.constants import SEED, REMOVE_LIST, CLEAN_DATA_PATH, SPLIT_DATA_PATH, SPLIT_TEST_SIZE

out_data_path = SPLIT_DATA_PATH

def date_split(df, target, split_date='1980-01-01'):
    '''
    Splitting approach borrowed from: EDA_Spike/Part 4_Model_v1.ipynb
    However, since each record stands on its own (due to lagging features) it is concievable to randomize the split (could even stratify for this)
    '''
    #df = df
    # split based on date
    df_train, df_test = df[df['Date'] < split_date], df[df['Date'] >= split_date]
    # split training data
    X_train = df_train.drop(REMOVE_LIST, axis=1)
    y_train = df_train[[target]]
    # split test data
    X_test = df_test.drop(REMOVE_LIST, axis=1)
    y_test = df_test[[target]]
    return X_train, y_train, X_test, y_test

def standard_split(df, target):
    X = df.drop(REMOVE_LIST, axis=1)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(    
        X, y, test_size=SPLIT_TEST_SIZE, stratify=y, random_state=SEED)
    
    return X_train, y_train, X_test, y_test


def set_drop_columns(target, indf):
    drop_roots = []
    if target == 'bear':
        print("Dropping S&P and USREC columns")
        drop_roots = ['USREC', 'S&P500 Price - Inflation Adjusted',
            'S&P500 Dividend Yield',
            'S&P500 PE ratio',
            'S&P500 Earnings Yield']
        
    elif target == 'Regime':
        print("Dropping USREC columns")
        drop_roots = ['USREC']
        
    else:
        print("No columns dropped")
        
    drop = []
    for root in drop_roots:
        # borrowed inspiration from: https://stackoverflow.com/questions/27275236/how-to-select-all-columns-whose-names-start-with-x-in-a-pandas-dataframe
        permutations =  [col for col in indf if col.startswith(root)]
        drop.extend(permutations)
        
    return drop

def split_and_save(df_features, split_fn, target, paths, no_smote_paths=None):
    '''
    Execute the provided split function and save each of the resultant dataframes
    '''
    
    drop_columns = set_drop_columns(target, df_features)
    
    df_features = df_features.drop(columns=drop_columns, errors='ignore').copy()    
    
    # split data
    X_train, y_train, X_test, y_test = split_fn(df_features, target)
    
    if no_smote_paths is not None:
        try:
            dfs = [X_train, y_train, X_test, y_test]
            for idx, df in enumerate(dfs):
                df.to_csv(no_smote_paths[idx], index=False)
        except Exception as ex:
            for path in no_smote_paths:
                if os.path.exists(path):
                    os.remove(path)
            raise ex
        
    
    # rebalance classes
    sm = SMOTE(random_state=SEED)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    X_train = X_res
    y_train = y_res

    # save
    try:
        dfs = [X_train, y_train, X_test, y_test]
        for idx, df in enumerate(dfs):
            df.to_csv(paths[idx], index=False)
    except Exception as ex:
        print('Failed to save training data')
        # print(ex)
        # clean up failed save
        for path in paths:
            if os.path.exists(path):
                os.remove(path)
        raise ex


def create_training_data(df_features, target):              
        
    # If we are using the 'stock-based indicator' as a target, remove these IDs (recession and stock indicators).
    if target == 'bear' or target == 'correction':        
        # Let's use a nice little trick from https://stackoverflow.com/questions/43822349/drop-column-that-starts-with 
        #   to remove any columns that starts with one of these (lag features will have same root!)
        for leaky_col in  ['USREC', 'SPASTT01USM657N', 'SPASTT01USM661N']:
            df_features = df_features.loc[:, ~df_features.columns.str.startswith(leaky_col)]    
    
    print(f"Creating date split training data for target {target}")    
    prefixes = ['X_train', 'y_train', 'X_test', 'y_test']
    names = [f'{p}_{target}' for p in prefixes]
        
    # ['X_train_date.csv', 'y_train_date.csv', 'X_test_date.csv', 'y_test_date.csv']
    date_names = [f'{name}_date.csv' for name in names]        
    date_names_no_smote = [f'{name}_date_no_smote.csv' for name in names]        
    date_paths = [SPLIT_DATA_PATH + fname for fname in date_names]
    date_paths_no_smote = [SPLIT_DATA_PATH + fname for fname in date_names_no_smote]
    if not os.path.exists(date_paths[0]):
        split_and_save(df_features, date_split, target, date_paths, date_paths_no_smote)

    print(f"Creating standard split training data for target {target}")
    # ['X_train_std.csv', 'y_train_std.csv', 'X_test_std.csv', 'y_test_std.csv']
    standard_fnames = [f'{name}_std.csv' for name in names]    
    standard_paths = [SPLIT_DATA_PATH + fname for fname in standard_fnames]
    if not os.path.exists(standard_paths[0]):
        split_and_save(df_features, standard_split, target, standard_paths)


def get_df(path):
    return pd.read_csv(path, index_col=0, parse_dates=True).reset_index(names='Date')


def split_data(split_target):
    '''
    Data pipeline based on work done for Milestone 1: https://github.com/jonkerr/SIADS593
    '''
    if not os.path.isdir(SPLIT_DATA_PATH):
        os.mkdir(SPLIT_DATA_PATH)

    df = get_df(CLEAN_DATA_PATH + 'merged.csv')

    # we'll get rid of this one eventually
#    if split_target in ['original', 'all']:
#        create_training_data(df)
    if split_target in ['bear', 'all']:
        create_training_data(df, 'bear')
    if split_target in ['rec', 'all']:
        create_training_data(df, 'Regime')
#    if split_target in ['corr', 'all']:
#        create_training_data(df, 'correction')
       
       
       
    """
    if split_option in ['lasso', 'all']:
        df = get_df(FEATURE_DATA_PATH + 'features_lasso.csv')
        create_training_data(df, 'lasso')
        
    if split_option in ['pca', 'all']:
        df = get_df(FEATURE_DATA_PATH + 'features_pca.csv')
        create_training_data(df, 'pca')
    """
    

'''
Handle command line arguments
'''
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    """
    parser.add_argument('-so', '--split_option',
                        help='Which split option? [lasso|pca|original|all] Default is all',
                        default="all",
                        required=False)
    """    
    parser.add_argument('-st', '--split_target',
                        help='Which split target? [bear|rec|all] Default is all',
                        default="all",
                        required=False)
    args = parser.parse_args()
    split_data(args.split_target)


