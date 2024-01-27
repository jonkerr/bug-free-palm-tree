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
    def __init__(self, filename) -> None:
        super().__init__()
        self.path = CLEAN_DATA_PATH + filename

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
            print('Failed to clean ', self.path)
            print(ex)
            # clean up failed clean
            if os.path.exists(self.path):
                os.remove(self.path)   

    @abstractmethod
    def _clean(self):
      pass


class CleanSPY(EfficientDataCleaner):
    '''
    Download all relevant FRED data and save in CSV format
    Based on some work in a Python notebook that Naomi created
    '''
    def __init__(self, filename='sp500_clean.csv') -> None:
       super().__init__(filename)

    def _clean(self):
        sp = pd.read_csv(RAW_DATA_PATH + 'sp500.csv', index_col = 0, parse_dates=True)
        # resample daily prices to monthly prices
        sp = sp.resample('MS').first()
        sp.to_csv(self.path)





'''
Data pipeline based on work done for Milestone 1: https://github.com/jonkerr/SIADS593
'''

def clean_data(clean_option):
    if clean_option in ['spy', 'all']:
        CleanSPY().clean()


'''
Handle command line arguments
'''
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # pass an arg using either "-do" or "--download_option"
    parser.add_argument('-co', '--clean_option',
                        help='Which file to clean? [spy] Default is all',
                        default="all",
                        required=False)
    args = parser.parse_args()
    clean_data(args.clean_option)
