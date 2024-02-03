import hidden
import time
import pandas as pd
import os
from abc import ABC, abstractmethod
from utils.multpl_data_scraper import MultplDataScraper

# configure data directory
DATA_PATH = './raw_data/'
if not os.path.isdir(DATA_PATH):
    os.mkdir(DATA_PATH)

"""
# FRED API (Previously gathered in Jupyter notebooks put together by Naomi.  I've just moved it into Python files for convenience and automation.)

Request API key:
https://fredaccount.stlouisfed.org/login/secure/

API documentation:
https://fred.stlouisfed.org/docs/api/fred/  

Tutorial: https://mortada.net/python-api-for-fred.html

The FRED API has a limit of 1000 items per request, with a rate limit of 120 requests per minute. If you receive a timeout error, please wait a while before downloading.
"""
from fredapi import Fred 
def get_fred_api():
    '''
    Use the hidden file to acquire the Fred API secrets
    ------
    Returns a python Fred APi wrapper    
    '''
    secrets = hidden.fred_secrets()
    return Fred(api_key=secrets['api_key'])
fred = get_fred_api()


def get_meta_data(search='United States', limit = 5000, order_by = 'popularity', freq='Monthly', sa=False):
  '''
  Returns the series ID and detailed information
  of economic indicators for the selected country from the FRED database.
  ----------
  Parameters
  ----------
  country: country name (str) or search keyword (e.g.,'GDP')
  order_by: valid options are 'popularity', 'search_rank', 'series_id', 'title', etc.
  freq: frequency ('Daily', 'Monthly', 'Quarterly', 'Annual' )
  sa: seasonally adjusted or not 
  '''
  data = fred.search(search)
  data = data[(data['frequency']==freq)]
  if sa == True:
    data = data[data['seasonal_adjustment'] !='Not Seasonally Adjusted']
  time.sleep(60)
  return data


class EfficientDownloader(ABC):
    def __init__(self, out_file) -> None:
        super().__init__()
        self.path = DATA_PATH + out_file
        self.observation_start='1/1/1871'

    def get_data(self):
        '''
        Only download if the file doesn't already exist
        Delete file if exception during download()
        '''
        if os.path.exists(self.path):
           return      
        try:  
            self._download()
        except Exception as ex:
            print('Failed to get ', self.path)
            print(ex)
            # clean up failed write
            if os.path.exists(self.path):
                os.remove(self.path)   

    @abstractmethod
    def _download(self):
      pass
   

class RecessionData(EfficientDownloader):
    '''
    Download all relevant FRED data and save in CSV format
    Based on some work in a Python notebook that Naomi created
    '''
    def __init__(self, out_file='recession.csv') -> None:
       super().__init__(out_file)

    def _download(self):
        print('> Getting recession data')
        recession = fred.get_series('USREC', observation_start=self.observation_start)
        recession.to_csv(self.path, index_label='Date', header=['Regime'])


class FredSeriesData(EfficientDownloader):
    '''
    Returns a dataframe with time series data retrieved from the FRED database
    Based on some work in a Python notebook that Naomi created
    ----------
    Parameters
    ----------
    id : FRED series id (list or array)
    observation_start = 'MM/DD/YYYY'
    '''
    def __init__(self, ids, out_file='econ_fred.csv', observation_start=None) -> None:
       super().__init__(out_file)
       self.ids = ids
       self.observation_start = observation_start or self.observation_start

    def _download(self):
        print('> Getting FRED data')
        dataset = {}
        start_time = time.time()
        for count, i in enumerate(self.ids):
            if count % 50 == 0:
                print(count, ' months downloaded')
            try:
                dataset[i] = fred.get_series(i, observation_start=self.observation_start)
            except Exception as ex:
                print(ex)
                print(i)
                time.sleep(60)

        df = pd.DataFrame(dataset)        
        df.to_csv(self.path)
        print(f'Download completed in {round((time.time() - start_time) / 60, 2)} minutes.')
        print(f'nubmer of indicators: ', df.shape[1])
        print(f'number of months: ', df.shape[0])        

    
class EconomicData(EfficientDownloader):
    '''
    Download historical economic data from multpl
    '''
    def __init__(self, out_file='econ_multpl.csv', observation_start=None) -> None:
       super().__init__(out_file)
       self.observation_start = observation_start or self.observation_start

    def _download(self):
        '''
        Scrape the desired data from www.multpl.com
        Based on Naomi's work in S&P500_data_v1.ipynb
        '''
        print('> Getting multpl economic data')
        df = MultplDataScraper().get_data()
        
        # prior to saving, we may want to restrict to dates as of observation_start
        df.to_csv(self.path)


'''
Data pipeline based on work done for Milestone 1: https://github.com/jonkerr/SIADS593
'''
def download(download_option):
    if download_option in ['rec', 'all']:
        RecessionData().get_data()
    if download_option in ['series', 'all']:
        # Naomi, we need some discussion on where ids.csv comes from 
        df_ids = pd.read_csv('curated_data/ids.csv')
        FredSeriesData(df_ids.id).get_data()
    if download_option in ['econ', 'all']:
        EconomicData().get_data()


'''
Handle command line arguments
'''
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # pass an arg using either "-do" or "--download_option"
    parser.add_argument('-do', '--download_option',
                        help='Which file to download? [rec|series|econ] Default is all',
                        default="all",
                        required=False)
    args = parser.parse_args()
    download(args.download_option)
