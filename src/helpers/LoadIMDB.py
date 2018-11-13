# Module to retrieve and/or build pickled DataFrames

import pandas as pd
import os # For working with file directory.

DEFAULT_PICKLE_LOCATION = './data/pickled/'
DEFAULT_DATA_DIRECTORY = './data/aclImdb/'
PRINT_DEBUG = False

def get_IMDB(train=True, positive=True):
    '''
    Get the DataFrame(DF) for specified training/test positive/negative data
    samples. Generates and save pickled DF if doens't exists.

    Params:
        train -- get the training/test set (default=True)
        positive -- get the positive/negative set (default=True)

    Return:
        A dataframe containing data from the IMDB dataset.
        Column = ['review_id', 'rating', 'review']
    '''
    train_set = 'train' if train else 'test'
    positive_set = 'pos' if positive else 'neg'
    pickle_filename = 'imdb_' + train_set + '_' + positive_set

    # Data directory
    generated_directory = os.path.join(DEFAULT_DATA_DIRECTORY,
            train_set, positive_set)

    # Where the pickle file will go.
    generated_pickle_file = os.path.join(DEFAULT_PICKLE_LOCATION,
            pickle_filename)

    # Check if there is already a pickle. If there is already a pickle just
    # load it.
    if (os.path.isfile(generated_pickle_file)):
        if (PRINT_DEBUG):
            print('loading', generated_pickle_file)
        return pd.read_pickle(generated_pickle_file)

    # Start buildind the DF
    if (PRINT_DEBUG):
        print('building', generated_pickle_file)

    # Build empty DF. This is the returned DF.
    imdb_df = pd.DataFrame(columns=['review_id', 'rating', 'review', ])
    number_of_items = 0; # Keep track of where to insert next item.

    # Get all the files in the directory
    for filename in os.listdir(generated_directory):

        # Get the review_id
        review_id = filename.split('_')[0]

        # Get the rating
        rating = filename.split('_')[1].split('.')[0]

        # Get the review from the file.
        with open(os.path.join(generated_directory, filename), 'rb') as f:
            review = f.read().strip().decode()
            imdb_df.loc[number_of_items] = [review_id, rating, review]
            number_of_items += 1

    imdb_df.to_pickle(generated_pickle_file)
    return imdb_df

# if __name__ == '__main__':
#     # Build all the dataframes.
#     get_IMDB(train=True, positive=True)
#     get_IMDB(train=True, positive=False)
#     get_IMDB(train=False, positive=True)
#     get_IMDB(train=False, positive=False)
