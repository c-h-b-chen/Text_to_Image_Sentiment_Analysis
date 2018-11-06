import pandas as pd

def get_YELP():
    '''
    Get a DataFrame of the yelp dataset.

    Return:
        Returns a DF containing yelp review data.
    '''
    yelp_dir = '../data/yelp/yelp.csv' 
    return pd.read_csv(yelp_dir)


