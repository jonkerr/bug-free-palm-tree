def calculate_bear_market(indf, price_col, include_corrections=True):
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
    ## max drawdown for a given period is over 20% AND
    ## cumulative drawdown hasn't hit the bottom yet
    df['bear'] = (df['ddmax'] < -0.2) & (df['ddmax'] < df.groupby('ddn')['dd'].transform('cummin'))
    
    # group bear markets into start/end periods (min/max dates)
    df['bearn'] = ((df['bear'] == True) & (df['bear'].shift() == False)).cumsum()

    # calculate the start and end dates of the bear market
    bears = df.reset_index().query('bear == True').groupby('bearn')['Date'].agg(['min', 'max'])

    # let's also consider corrections
    corrections = None
    if include_corrections:
        df['correction'] = (~df['bear']) & (df['ddmax'] < -0.1) & (df['ddmax'] < df.groupby('ddn')['dd'].transform('cummin'))
        df['corrn'] = ((df['correction'] == True) & (df['correction'].shift() == False)).cumsum()
        corrections = df.reset_index().query('correction == True').groupby('corrn')['Date'].agg(['min', 'max'])
        df = df.drop(columns=['corrn'])

    '''
    Now that we've identified if we're in a bear market, ignore the following features as they 
    are not relevant for training:
    period numbering:  ddn, bearn
    only meaninful during a bear market: ddmax
    It might be meaningful to keep ddn and use either bear or correction as possible targets.
    '''    
    df = df.drop(columns=['ddn','bearn','ddmax'])
    
    return df, bears, corrections


def add_pct_change(df):
    '''
    Calculate the % change from the previous month.
    A couple limitations of this approach:
    1. We assume the 20% drop (signaling a bear market) is month to month.  It says nothing about droping within a month and recoverying to > -20%, which would hide the signal.
    2. If it takes more than one month.  e.g. market drops 10% or more for two consecutive months.  We wouldn't be able to detect that. 
    '''
    col_sp = 'S&P500 Price - Inflation Adjusted'
    df['sp_pct_change'] = df[col_sp].pct_change(fill_method=None)
    df['sp_pct_change'] = df['pct_change'].apply(lambda x: round(x*100,2))
    return df