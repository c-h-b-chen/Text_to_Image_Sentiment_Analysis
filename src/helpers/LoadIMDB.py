# Module to retrieve and/or build pickled DataFrames

import numpy as np
import gensim
import pandas as pd
import os # For working with file directory.

import LoadEmbeddings

DEFAULT_PICKLE_LOCATION = '../data/pickled/'
DEFAULT_DATA_DIRECTORY = '../data/aclImdb/'
PRINT_DEBUG = True
NUM_WORDS = 299

W2V_M1_LOC = "../data/word2vec_models/w2v_m1.model"
W2V_M2_LOC = "../data/word2vec_models/w2v_m2.model"
W2V_M3_LOC = "../data/word2vec_models/w2v_m3.model"

ITEMS_TO_REMOVE = ['<br />', '.', ',', '*', '-', '%', '$', ';', '=', '[', ']',
        '(', ')', '&', '#', '@']

def gen_image_format(review, wv_m1, wv_m2, wv_m3):
    emb_review = []
    for wv in [wv_m1, wv_m2, wv_m3]:
        emb_review.append([wv.vocab[word].index for word in review.split()])
    return emb_review[:NUM_WORDS]

def convert_rating(rating):
    ''' Helper function to convert 'rating' column of the dataframes to 1 of 4 
    values
    '''
    if rating < 3:
        return 0
    elif rating < 5:
        return 1
    elif rating < 9:
        return 2
    else:
        return 3

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
    imdb_df = pd.DataFrame(columns=['review_id', 'rating', 'review'])
    number_of_items = 0; # Keep track of where to insert next item.

    # Get all the files in the directory
    for filename in os.listdir(generated_directory):

        # Get the review_id
        review_id = filename.split('_')[0]

        # Get the rating, convert to 1 of 4 values.
        rating = filename.split('_')[1].split('.')[0]
        rating = convert_rating(int(rating))

        # Get the review from the file.
        with open(os.path.join(generated_directory, filename), 'rb') as f:
            review = f.read().strip().decode()
            review = review.encode('ascii', errors='ignore').decode()

            # Remove all the extra sub_strings/characters we don't want.
            for sub_string in ITEMS_TO_REMOVE:
                review = review.replace(sub_string, ' ')
            review = review.lower() # Make everything lowercase too.
            review = str(review)

            imdb_df.loc[number_of_items] = [review_id, rating, review]
            number_of_items += 1

    imdb_df.to_pickle(generated_pickle_file)
    return imdb_df

def get_emb_IMDB(w2v_m1_loc=W2V_M1_LOC, w2v_m2_loc=W2V_M2_LOC,
        w2v_m3_loc=W2V_M3_LOC, train=True, positive=False, num_words=NUM_WORDS):
    ''' Get the embeding equivalent of the dataset.
    params:
        w2v_m1 -- First wordvector model.
        w2v_m2 -- Second wordvector model.
        w2v_m3 -- Third wordvector model.
    '''
    # Use the word vectors to make an encoding of the vectors.
    train_set = 'train' if train else 'test'
    positive_set = 'pos' if positive else 'neg'
    pickle_filename = 'imdb_emb_' + train_set + '_' + positive_set

    file_dir = os.path.join(DEFAULT_PICKLE_LOCATION, pickle_filename)

    # Check if we already have the file pickled.
    if (os.path.isfile(file_dir)):
        if (PRINT_DEBUG):
            print('loading', file_dir)
        return pd.read_pickle(file_dir)

    print("Buildilng", file_dir)

    wv_m1 = LoadEmbeddings.get_wordvec(w2v_m1_loc)
    wv_m2 = LoadEmbeddings.get_wordvec(w2v_m2_loc)
    wv_m3 = LoadEmbeddings.get_wordvec(w2v_m3_loc)

#    wv_m1 = gensim.models.KeyedVectors.load(w2v_m1_loc, mmap='r')
#    wv_m2 = gensim.models.KeyedVectors.load(w2v_m2_loc, mmap='r')
#    wv_m3 = gensim.models.KeyedVectors.load(w2v_m3_loc, mmap='r')

    # Don't have a pickled file. Make it

    data = get_IMDB(train=train, positive=positive)
    data['embedding'] = [gen_image_format(sub, wv_m1, wv_m2, wv_m3) for sub in
        data['review']]
    data.to_pickle(file_dir)
    return data

def dataset_IMDB(train=True):
    # Get the positive and negative datasets
    pos = get_emb_IMDB(train=train, positive=True)
    neg = get_emb_IMDB(train=train, positive=False)

    # combine the dataset, shuffle the data.
    data = pd.concat([pos, neg])
    data = data.sample(frac=1)
    return data
    

if __name__ == '__main__':
    # Build all the normal dataframes.
    get_IMDB(train=True, positive=True)
    get_IMDB(train=True, positive=False)
    get_IMDB(train=False, positive=True)
    get_IMDB(train=False, positive=False)

    get_emb_IMDB(train=True, positive=True)
    get_emb_IMDB(train=True, positive=False)
    get_emb_IMDB(train=False, positive=True)
    get_emb_IMDB(train=False, positive=False)


