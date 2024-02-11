'''
Take raw data and perform any required processing steps.
Output is a set of training data and a set of test data.
'''

# data pipeline based on work done for Milestone 1: https://github.com/jonkerr/SIADS593
import pandas as pd
import os
from abc import ABC, abstractmethod
#import utils.cleaning_utils as cu

# utils
from utils.decorators import file_check_decorator
from utils.constants import *
from utils.feature_engineering import *

# set data path for wrappers
out_data_path = CLEAN_DATA_PATH


@file_check_decorator(out_data_path)
def clean_multpl(out_file, in_file):
    '''
    Adds bear market calculation.
    '''
    df = pd.read_csv(RAW_DATA_PATH + in_file, index_col=0, parse_dates=True)

    # Yield is published the last day of the month.  Fill it to the first day so we don't lose it.
    div_yield = 'S&P500 Dividend Yield'
    df[div_yield] = df[div_yield].ffill()
    # still need to figure out what to do with 'S&P500 Earnings'

    # drop dates that aren't on the first of the month
    df = df[df.index.day == 1]
    # add features but only unpack the first item (the dataframe)
    df, bears, corrections = calculate_bear_market(
        df, price_col='S&P500 Price - Inflation Adjusted')
    # df = add_pct_change(df)
    # save
    df.to_csv(out_file)
    return df


@file_check_decorator(out_data_path)
def merge_data(out_file):
        paths = [
            RAW_DATA_PATH + 'econ_fred.csv',
            CLEAN_DATA_PATH + 'multpl_clean.csv',
#            RAW_DATA_PATH + 'recession.csv'        
        ]
        # read and merge files (both use date as index)
        dfs = [pd.read_csv(path, index_col=0, parse_dates=True)
               for path in paths]

        df = pd.concat(dfs, axis=1)
        
        # we should decide between "S&P500 Price" and	"S&P500 Price - Inflation Adjusted"
        # using both would be colinear.  For now, let's use inflation adjusted only
        df = df.drop(columns=['S&P500 Price'])

        # add lag features
        df = add_lag_features(df)

        # determine start date to minimize na cols
        # this would be a key area to adjust to get different data sets
        df = df[df.index >= POST_CLEANING_START_DATE]

        # cols to explicitly keep (maybe ffil or something?)
        # keep = ['case Shiller Home Price Index','S&P500 Earnings']

        # remove empty columns.
        df, _ = remove_variables(df.copy(), n=10)

        # finally, drop na
        df.dropna(axis=0, inplace=True)
        
        # Iteratively difference the time series until the number of non-stationary columns is less than a specified threshold.
        df, _ = stationarize_data(df, threshold=0.01)
        
        # shift the bear markets back a month to promote trying to predict bear markets, one month in the future
        df['bear'] = df['bear'].shift(-1).ffill()
        
        # save
        df.to_csv(out_file)


'''
Data pipeline based on work done for Milestone 1: https://github.com/jonkerr/SIADS593
'''
def clean_data(clean_option):
    if not os.path.isdir(CLEAN_DATA_PATH):
        os.mkdir(CLEAN_DATA_PATH)
    
    if clean_option in ['multpl', 'all']:
        clean_multpl('multpl_clean.csv', 'econ_multpl.csv')
        #CleanMultpl().clean()

    if clean_option in ['merge', 'all']:
        merge_data('merged.csv')
        #MergeData().clean()

    if clean_option in ['train', 'all']:
        pass
        #TrainingData().create_training_data()


'''
Handle command line arguments
'''
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # pass an arg using either "-do" or "--download_option"
    parser.add_argument('-co', '--clean_option',
                        help='Which file to clean? [multpl|merge|train] Default is all',
                        default="all",
                        required=False)
    args = parser.parse_args()
    clean_data(args.clean_option)
