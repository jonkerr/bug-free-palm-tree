import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# project constants
from utils.constants import SEED, REMOVE_LIST, TARGET, CLEAN_DATA_PATH, TRAINING_DATA_PATH, FEATURE_DATA_PATH

out_data_path = TRAINING_DATA_PATH

#df_features_and_targets = pd.read_csv(CLEAN_DATA_PATH + 'merged.csv', index_col=0, parse_dates=True).reset_index(names='Date')

def date_split(df, split_date='1980-01-01'):
    '''
    Splitting approach borrowed from: EDA_Spike/Part 4_Model_v1.ipynb
    However, since each record stands on its own (due to lagging features) it is concievable to randomize the split (could even stratify for this)
    '''
    #df = df
    # split based on date
    df_train, df_test = df[df['Date'] < split_date], df[df['Date'] >= split_date]
    # split training data
    X_train = df_train.drop(REMOVE_LIST, axis=1)
    y_train = df_train[[TARGET]]
    # split test data
    X_test = df_test.drop(REMOVE_LIST, axis=1)
    y_test = df_test[[TARGET]]
    return X_train, y_train, X_test, y_test


def standard_split(df):
    X = df.drop(REMOVE_LIST, axis=1)
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(    
        X, y, test_size=0.25, stratify=y, random_state=SEED)
    return X_train, y_train, X_test, y_test


def split_and_save(df_features, split_fn, paths):
    '''
    Execute the provided split function and save each of the resultant dataframes
    '''
    # split data
    X_train, y_train, X_test, y_test = split_fn(df_features)
    # standardize
    #X_train, X_test = standardize(X_train, X_test)

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


def create_training_data(df_features, feature_selection_method=None):           
    print("Creating date split training data")
    
    prefixes = ['X_train', 'y_train', 'X_test', 'y_test']
    names = [f'{p}_{feature_selection_method}' if feature_selection_method else p  for p in prefixes]
        
    # ['X_train_date.csv', 'y_train_date.csv', 'X_test_date.csv', 'y_test_date.csv']
    date_names = [f'{name}_date.csv' for name in names]        
    date_paths = [TRAINING_DATA_PATH + fname for fname in date_names]
    if not os.path.exists(date_paths[0]):
        split_and_save(df_features, date_split, date_paths)

    print("Creating standard split training data")
    # ['X_train_std.csv', 'y_train_std.csv', 'X_test_std.csv', 'y_test_std.csv']
    standard_fnames = [f'{name}_std.csv' for name in names]    
    standard_paths = [TRAINING_DATA_PATH + fname for fname in standard_fnames]
    if not os.path.exists(standard_paths[0]):
        split_and_save(df_features, standard_split, standard_paths)


def standardize(X_train, X_test):
    '''
    Use StandardScaler to scale the data for models that require it.

    Parameters:
    - X_train: The training features.
    - X_test: The test features. Optional if you only want to scale the training data.

    Returns:
    - X_train_scaled: The scaled training features as a dataframe.
    - X_test_scaled: The scaled test features as a dataframe.
    '''
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)

    # transform returns an np.array.  We need a dataframe.
    def to_df(df_scaled, df):
        return pd.DataFrame(df_scaled, index=df.index, columns=df.columns)

    return to_df(X_train_scaled, X_train), to_df(X_test_scaled, X_test)


def get_df(path):
    return pd.read_csv(path, index_col=0, parse_dates=True).reset_index(names='Date')


def split_data(split_option):
    '''
    Data pipeline based on work done for Milestone 1: https://github.com/jonkerr/SIADS593
    '''
    if not os.path.isdir(TRAINING_DATA_PATH):
        os.mkdir(TRAINING_DATA_PATH)

    # we'll get rid of this one eventually
    if split_option in ['original', 'all']:
        df = get_df(CLEAN_DATA_PATH + 'merged.csv')
        create_training_data(df)
       
    if split_option in ['lasso', 'all']:
        df = get_df(FEATURE_DATA_PATH + 'features_lasso.csv')
        create_training_data(df, 'lasso')
        
    if split_option in ['pca', 'all']:
        df = get_df(FEATURE_DATA_PATH + 'features_pca.csv')
        create_training_data(df, 'PCA')
    

'''
Handle command line arguments
'''
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # pass an arg using either "-do" or "--download_option"
    parser.add_argument('-so', '--split_option',
                        help='Which split option? [lasso|pca|original|all] Default is all',
                        default="all",
                        required=False)
    args = parser.parse_args()
    split_data(args.split_option)


