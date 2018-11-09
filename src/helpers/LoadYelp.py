import pandas as pd

PRINT_DEBUG = False

def get_YELP():
    '''
    Get a DataFrame of the yelp dataset.

    Return:
        Returns a DF containing yelp review data.
    '''
    yelp_dir = '../data/yelp/yelp.csv' 
    if PRINT_DEBUG:
        print('loading yelp dataset ')
    return pd.read_csv(yelp_dir)

if __name__ == '__main__':
    PRINT_DEBUG = True
    get_YELP()

