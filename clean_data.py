'''
Take raw data and perform any required processing steps.
Output is a set of training data and a set of test data.
'''

# data pipeline based on work done for Milestone 1: https://github.com/jonkerr/SIADS593
import pandas as pd
import os
from abc import ABC, abstractmethod

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


class CleanFRED(EfficientDataCleaner):
    '''
    Clean the economic data downloaded from FRED
    '''
    def __init__(self, out_file='fred_clean.csv') -> None:
       super().__init__(out_file)
       
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
        df = pd.read_csv(RAW_DATA_PATH + self.in_file, index_col = 0, parse_dates=True)
        # drop dates that aren't on the first of the month
        df = df[df.index.day == 1]
        df.to_csv(self.path)
        



class CleanSPY(EfficientDataCleaner):
    '''
    Download all relevant FRED data and save in CSV format
    Based on some work in a Python notebook that Naomi created
    '''
    def __init__(self, filename='sp500_clean.csv') -> None:
       super().__init__(filename)

    def _clean(self):
        sp = pd.read_csv(RAW_DATA_PATH + 'sp500.csv', index_col = 0, parse_dates=True)
        sp.columns = ['values']
        # add some additional features
        sp = self.add_std_features(sp)
        # resample daily prices to monthly prices
        sp = sp.resample('MS').first()
        # get month to month pct change
        sp = self.add_pct_change(sp)
        sp.to_csv(self.path)

    def add_std_features(self, indf):
        '''
        I'm not sure if we'll keep this or not but thought it might be interesting to see if the standard deviation
        of daily price movement (across last 30 days or a month) gave any insight on our target variable.
        '''
        df = indf.copy()
        # group by month and year and take the std
        df_std_m = df.groupby([(df.index.year), (df.index.month)]).std().reset_index()
        df_std_m.columns = ['year','month','std_month']
        # merge
        df['year'] = df.index.year
        df['month'] = df.index.month
        df = df.reset_index()
        df = df.merge(df_std_m, on=['year', 'month']).set_index('index')
        # std for last 30 days (for comparison)
        df['std_30'] = df['values'].rolling(30, min_periods=3).std()
        # take mean across different versions of std
        df['std_mean'] = df[['std_month','std_30']].mean(axis=1)
        return df.drop(columns=['year','month']).round({'std_month':2, 'std_30':2, 'std_mean':2})
    
    def add_pct_change(self, df):
        '''
        Calculate the % change from the previous month.
        A couple limitations of this approach:
        1. We assume the 20% drop (signaling a bear market) is month to month.  It says nothing about droping within a month and recoverying to > -20%, which would hide the signal.
        2. If it takes more than one month.  e.g. market drops 10% or more for two consecutive months.  We wouldn't be able to detect that. 
        '''
        df['pct_change'] = df['values'].pct_change(fill_method=None)
        df['pct_change'] = df['pct_change'].apply(lambda x: round(x*100,2))
        return df



'''
Data pipeline based on work done for Milestone 1: https://github.com/jonkerr/SIADS593
'''

def clean_data(clean_option):
    if clean_option in ['fred', 'all']:
        CleanFRED().clean()

    if clean_option in ['multpl', 'all']:
        CleanMultpl().clean()
        
    if clean_option in ['spy', 'all']:
        pass
        


'''
Handle command line arguments
'''
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # pass an arg using either "-do" or "--download_option"
    parser.add_argument('-co', '--clean_option',
                        help='Which file to clean? [spy|multpl|fred] Default is all',
                        default="all",
                        required=False)
    args = parser.parse_args()
    clean_data(args.clean_option)
