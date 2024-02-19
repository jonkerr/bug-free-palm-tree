import hidden
import time
import pandas as pd
import os
from utils.constants import *
from utils.multpl_data_scraper import MultplDataScraper
from collections import defaultdict
from fredapi import Fred

# utils
from utils.decorators import file_check_decorator
from utils.constants import *

out_data_path = RAW_DATA_PATH

"""
# FRED API (Previously gathered in Jupyter notebooks put together by Naomi.  I've just moved it into Python files for convenience and automation.)

Request API key:
https://fredaccount.stlouisfed.org/login/secure/

API documentation:
https://fred.stlouisfed.org/docs/api/fred/  

Tutorial: https://mortada.net/python-api-for-fred.html

The FRED API has a limit of 1000 items per request, with a rate limit of 120 requests per minute. If you receive a timeout error, please wait a while before downloading.
"""
def get_fred_api():
    '''
    Use the hidden file to acquire the Fred API secrets
    ------
    Returns a python Fred APi wrapper    
    '''
    secrets = hidden.fred_secrets()
    return Fred(api_key=secrets['api_key'])


fred = get_fred_api()

def get_meta_data(search='United States', limit=5000, order_by='popularity', freq='Monthly', sa=False):
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
    data = data[(data['frequency'] == freq)]
    if sa == True:
        data = data[data['seasonal_adjustment'] != 'Not Seasonally Adjusted']
    time.sleep(60)
    return data


@file_check_decorator(out_data_path)
def get_recession_data(out_file, observation_start=DEFAULT_OBSERVATION_START):
    '''
    Download all relevant FRED data and save in CSV format
    Based on some work in a Python notebook that Naomi created
    '''
    print('> Getting recession data')
    recession = fred.get_series('USREC', observation_start=observation_start)
    recession.to_csv(out_file, index_label='Date', header=['Regime'])
    return recession


@file_check_decorator(out_data_path)
def get_fred_series_data(out_file, ids, observation_start=None):
    '''
    Returns a dataframe with time series data retrieved from the FRED database
    Based on some work in a Python notebook that Naomi created
    ----------
    Parameters
    ----------
    out_file: name of file to store the downloaded data
    id : FRED series id (list or array)
    observation_start = 'MM/DD/YYYY'
    '''
    print('> Getting FRED data')
    dataset = {}
    start_time = time.time()

    for count, i in enumerate(ids):
        if count % 50 == 0:
            print(count, ' months downloaded')
        try:
            dataset[i] = fred.get_series(
                i, observation_start=observation_start)
        except Exception as ex:
            print(ex)
            print(i)
            time.sleep(60)

    df = pd.DataFrame(dataset)
    df.to_csv(out_file)
    print(
        f'Download completed in {round((time.time() - start_time) / 60, 2)} minutes.')
    print(f'nubmer of indicators: ', df.shape[1])
    print(f'number of months: ', df.shape[0])
    return df


@file_check_decorator(out_data_path)
def get_multpl_data(out_file):
    '''
    Scrape the desired data from www.multpl.com
    Based on Naomi's work in S&P500_data_v1.ipynb
    '''
    print('> Getting multpl economic data')
    df = MultplDataScraper().get_data()

    # prior to saving, we may want to restrict to dates as of observation_start
    df.to_csv(out_file)
    return df


def download(download_option):
    '''
    Data pipeline based on work done for Milestone 1: https://github.com/jonkerr/SIADS593
    '''
    if not os.path.isdir(RAW_DATA_PATH):
        os.mkdir(RAW_DATA_PATH)

    data = defaultdict(None)
    if download_option in ['rec', 'all']:
        data['recession'] = get_recession_data('recession.csv')
    if download_option in ['series', 'all']:
        # Naomi, we need some discussion on where ids.csv comes from
        data['ids'] = pd.read_csv('curated_data/ids.csv')
        get_fred_series_data('econ_fred.csv', data['ids'].id, DEFAULT_OBSERVATION_START)
    if download_option in ['econ', 'all']:
        data['multpl'] = get_multpl_data('econ_multpl.csv')


'''
Handle command line arguments
'''
if __name__ == '__main__':   
    import argparse
    parser = argparse.ArgumentParser()
    # pass an arg using either "-do" or "--download_option"
    parser.add_argument('-do', '--download_option',
                        help='Which file to download? [rec|series|econ|rd] Default is all',
                        default="all",
                        required=False)
    args = parser.parse_args()
    download(args.download_option)
