
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler

from utils.constants import CANDIDATE_TARGETS

def remove_variables(df, n=10, keep=None):
    '''
    From: EDA_Spike/Part 2_Data Cleaning_v1.ipynb
    ----------
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
    return df.copy(), dropped_cols


def add_lag_features(df, lags=[3, 6, 9, 12, 18]):
    '''
    From: EDA_Spike/Part 3_Data Processing_v1.ipynb
    ----------
    Returns df with the specified lag variables added
    ----------
    Parameters
    ----------
    df : original dataframe
    lags: list of months to include lags
    '''
    lag_cols = {}
    for col in df.drop(columns=CANDIDATE_TARGETS):
        for n in lags:
            lag_cols['{}_{}M_lag'.format(col, n)] = df[col].shift(
                n).ffill().values

    '''
    lags.insert(0,2)
    lags.insert(0,1)
    
    def pct_change(base_col, next_col):
        return (next_col.sub(base_col)) / base_col * 100
    
    for n in lags:
        col = 'S&P500 Price - Inflation Adjusted'            
        next_col = df[col].shift(n).ffill()            
        lag_cols['{}_pct_change_{}M_lag'.format(col, n)] = pct_change(col, next_col)            
    '''

    new_df = pd.DataFrame(lag_cols, index=df.index)
    df = pd.concat([df, new_df], axis=1)
    return df


def stationarize_data(df, threshold=0.01, max_non_stationary_cols=10):
    '''
    From: EDA_Spike/Part 3_Data Processing_v1.ipynb
    ----------
    Iteratively difference the time series until the number of non-stationary columns is less than a specified threshold.

    Parameters:
    - df (pd.DataFrame): Input dataframe containing time series data.
    - threshold (float): ADF test p-value threshold for considering a column as non-stationary.
    - max_non_stationary_cols (int): Maximum number of non-stationary columns allowed.

    Returns:
    - df (pd.DataFrame): Dataframe after differencing.
    - non_stationary_cols (list): List of non-stationary columns.
    '''
    # Initial set of non-stationary columns
    non_stationary_cols = df.drop(columns=CANDIDATE_TARGETS).columns
    iteration_count = 0

    # Iterate until the number of non-stationary columns is less than the specified threshold
    while len(non_stationary_cols) >= max_non_stationary_cols:
        iteration_count += 1
        print(f'\nIteration {iteration_count}:')

        # Columns to be differenced in this iteration
        need_diff = []

        # Check ADF test p-value for each column
        for col in df.drop(columns=CANDIDATE_TARGETS).columns:
            result = adfuller(df[col])
            if result[1] > threshold:
                need_diff.append(col)
                df[col] = df[col].diff()

        # Update the set of non-stationary columns for the next iteration
        non_stationary_cols = need_diff

        # Drop NaN rows after differencing
        df.dropna(axis=0, inplace=True)
        print(
            f'Number of Non-stationary columns: {len(non_stationary_cols)}')

    print(f'\nDataframe shape after differencing: {df.shape}')
    return df.copy(), non_stationary_cols


def calculate_bear_market(indf, price_col, include_corrections=True):
    '''
    Calculate a bear market.
    Adapted from https://stackoverflow.com/questions/64830383/calculating-bull-bear-markets-in-pandas
    -----
    Note:   The original algorithm takes a different apprach than we ultimately plan to.
            They calculate a bear market as a 20% decline from an ABSOLUTE market peak.
            We intend to calculate a bear market as a 20% decline from an RELATIVE market peak.

            Our approach is more difficult and potentially susceptible to bias as we'd be making a (human judgement based) decision
            on how far back to look to find the most recent peak.  That said, we feel it's worth the risk as the current model
            catagorizes the period between 2000-2009 as a single bear market.  An investor following this model would have missed
            the (secular?) bull market from 2003-2008.  As such, we'd prefer to categorize this period as two bear markets
            2000-2003 and 2008-2009.
    '''
    # Once implemented, use the relative peak approach instead of the absolute peak
    # return calculate_bear_market_relative_peak(indf, price_col, include_corrections)
    return calculate_bear_market_absolute_peak(indf, price_col, include_corrections)


def calculate_bear_market_absolute_peak(indf, price_col, include_corrections=True):
    '''
    Calculate a bear market.
    Adapted from https://stackoverflow.com/questions/64830383/calculating-bull-bear-markets-in-pandas
    I've added comments as I've reverse engineered it
    '''
    # avoid directly modifying the df (in case that's not desired)
    df = indf.copy()

    # get the % drawdown from the high up to that point ( cummax() )
    # this number is negative if lower or 0 if new high
    df['dd'] = df[price_col].div(df[price_col].cummax()).sub(1)

    # if current reading is immediately lower after a new high, add a new group number (ddn)
    df['ddn'] = ((df['dd'] < 0.) & (df['dd'].shift() == 0.)).cumsum()

    # get the largest drawdown for the group.  e.g. market bottom
    df['ddmax'] = df.groupby('ddn')['dd'].transform('min')

    # determine if this is a bear market if both conditions are true:
    # max drawdown for a given period is over 20% AND
    # cumulative drawdown hasn't hit the bottom yet
    df['bear'] = (df['ddmax'] < -0.2) & (df['ddmax'] <
                                         df.groupby('ddn')['dd'].transform('cummin'))

    # group bear markets into start/end periods (min/max dates)
    df['bearn'] = ((df['bear'] == True) & (
        df['bear'].shift() == False)).cumsum()

    # calculate the start and end dates of the bear market
    bears = df.reset_index().query('bear == True').groupby('bearn')[
        'Date'].agg(['min', 'max'])

    # let's also consider corrections
    corrections = None
    if include_corrections:
        df['correction'] = (~df['bear']) & (
            df['ddmax'] < -0.1) & (df['ddmax'] < df.groupby('ddn')['dd'].transform('cummin'))
        df['corrn'] = ((df['correction'] == True) & (
            df['correction'].shift() == False)).cumsum()
        corrections = df.reset_index().query('correction == True').groupby('corrn')[
            'Date'].agg(['min', 'max'])
        df = df.drop(columns=['corrn'])

    '''
    Now that we've identified if we're in a bear market, ignore the following features as they 
    are not relevant for training:
    period numbering:  ddn, bearn
    only meaninful during a bear market: ddmax
    It might be meaningful to keep ddn and use either bear or correction as possible targets.
    '''
    df = df.drop(columns=['dd', 'ddn', 'bearn', 'ddmax'])

    return df, bears, corrections


def calculate_bear_market_relative_peak(indf, price_col, include_corrections=True):
    '''
    Still needs to be implemented to detect shorter term bear markets/downturns that are not dependent on absolute peak
    '''
    pass


def add_pct_change(df, market_col='S&P500 Price - Inflation Adjusted'):
    '''
    Calculate the % change from the previous month.
    A couple limitations of this approach:
    1. We assume the 20% drop (signaling a bear market) is month to month.  It says nothing about droping within a month and recoverying to > -20%, which would hide the signal.
    2. If it takes more than one month.  e.g. market drops 10% or more for two consecutive months.  We wouldn't be able to detect that. 
    '''
    df['pct_change'] = df[market_col].pct_change(fill_method=None)
    df['pct_change'] = df['pct_change'].apply(lambda x: round(x*100, 2))
    return df
