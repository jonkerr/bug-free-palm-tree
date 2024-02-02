'''
Take raw data and perform any required processing steps.
Output is a set of training data and a set of test data.
'''

# data pipeline based on work done for Milestone 1: https://github.com/jonkerr/SIADS593
import pandas as pd
import os
from abc import ABC, abstractmethod
import utils.cleaning_utils as cu

# configure data directory
RAW_DATA_PATH = './raw_data/'
CLEAN_DATA_PATH = './clean_data/'
if not os.path.isdir(CLEAN_DATA_PATH):
    os.mkdir(CLEAN_DATA_PATH)


class EfficientDataCleaner(ABC):
    def __init__(self, out_file) -> None:
        super().__init__()
        self.path = CLEAN_DATA_PATH + out_file

    def clean(self):
        '''
        Only clean if the target file doesn't already exist
        Delete file if exception during clean()
        '''
        if os.path.exists(self.path):
            return
        try:
            self._clean()
        except Exception as ex:
            print('Failed to create ', self.path)
            print(ex)
            # clean up failed clean
            if os.path.exists(self.path):
                os.remove(self.path)

    @abstractmethod
    def _clean(self):
        pass


class CleanMultpl(EfficientDataCleaner):
    '''
    Clean the economic data downloaded from multpl
    '''
    def __init__(self, out_file='multpl_clean.csv', in_file='econ_multpl.csv') -> None:
        super().__init__(out_file)
        self.in_file = in_file

    def _clean(self):
        df = pd.read_csv(RAW_DATA_PATH + self.in_file,
                         index_col=0, parse_dates=True)

        # Yield is published the last day of the month.  Fill it to the first day so we don't lose it.
        div_yield = 'S&P500 Dividend Yield'
        df[div_yield] = df[div_yield].ffill()
        # still need to figure out what to do with 'S&P500 Earnings'

        # drop dates that aren't on the first of the month
        df = df[df.index.day == 1]
        # add features but only unpack the first item (the dataframe)
        df, bears, corrections = cu.calculate_bear_market(df, price_col = 'S&P500 Price - Inflation Adjusted')
        #df = cu.add_pct_change(df)
        # save
        df.to_csv(self.path)
        


class MergeData(EfficientDataCleaner):
    '''
    Merge multiple files and deal with nulls
    '''
    def __init__(self, out_file='merged.csv') -> None:
        super().__init__(out_file)

    def _clean(self):
        paths = [
            RAW_DATA_PATH + 'econ_fred.csv',
            CLEAN_DATA_PATH + 'multpl_clean.csv'
        ]
        dfs = [pd.read_csv(path, index_col=0, parse_dates=True)
               for path in paths]

        df = pd.concat(dfs, axis=1)
        
        # determine start date to minimize na cols
        df = df[df.index > '1919-01-01']
        
        # cols to explicitly keep (maybe ffil or something?)
        keep = ['case Shiller Home Price Index','S&P500 Earnings']
                
        # remove empty columns.  
        df, dropped = self.remove_variables(df, n=10, keep=keep)

        df.to_csv(self.path)
        
        
    def remove_variables(self, df, n=10, keep=None):
        '''
        From: Part 2_Data Cleaning_v1.ipynb
        Removes a variable if it has more than "n" NaN values
        Returns a dataframe without variables with NaNs
        ----------
        Parameters
        ----------
        df : dataframe
        n : number of NaN values (int)
        '''
        dropped_cols = {}
        for col in df.columns:
            if keep and col in keep:
                continue
            nas = df[col].isna().sum()
            if nas > n:
                dropped_cols[col] = nas
                df.drop(col, axis=1, inplace=True)
        return df, dropped_cols

'''
Data pipeline based on work done for Milestone 1: https://github.com/jonkerr/SIADS593
'''
def clean_data(clean_option):
    if clean_option in ['multpl', 'all']:
        CleanMultpl().clean()

    if clean_option in ['merge', 'all']:
        MergeData().clean()


'''
Handle command line arguments
'''
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # pass an arg using either "-do" or "--download_option"
    parser.add_argument('-co', '--clean_option',
                        help='Which file to clean? [multpl|merge] Default is all',
                        default="all",
                        required=False)
    args = parser.parse_args()
    clean_data(args.clean_option)
