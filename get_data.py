# data pipeline based on work done for Milestone 1: https://github.com/jonkerr/SIADS593

# configure data directory
DATA_PATH = './raw_data/'
import os
if not os.path.isdir(DATA_PATH):
    os.mkdir(DATA_PATH)


def get_fred_api():
    '''
    Use the hidden file to acquire the Fred API secrets
    ------
    Returns a python Fred APi wrapper    
    '''
    import hidden
    from fredapi import Fred 

    secrets = hidden.fred_secrets()
    return Fred(api_key=secrets['api_key'])
fred = get_fred_api()


def get_recession_data():
    '''
    Download all relevant FRED data and save in CSV format
    Based on some work in a Python notebook that Naomi created
    '''
    recession_csv = DATA_PATH + 'recession.csv'
    # only download if not already available
    if not os.path.exists(recession_csv):
        # Regime indicator: Recession = 1, Normal = 0
        recession = fred.get_series('USREC')
        recession.to_csv(recession_csv, index_label='Date', header=['Regime'])


def download_spy():
    pass
    ## need to standardize dates to match recession dates
    #1854-12-01
    #sp = fred.get_series('SP500', observation_start='1/31/2014')
    


def download(download_option):
    if download_option in ['rec', 'all']:
        get_recession_data()
    if download_option in ['spy', 'all']:
        download_spy()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # pass an arg using either "-do" or "--download_option"
    parser.add_argument('-do', '--download_option',
                        help='Which file to download? [rec|spy] Default is all',
                        default="all",
                        required=False)
    args = parser.parse_args()
    download(args.download_option)
