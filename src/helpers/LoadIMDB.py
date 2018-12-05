# Module to retrieve and/or build pickled DataFrames

import numpy as np
import gensim
import pandas as pd
import Settings
import os # For working with file directory.

import LoadEmbeddings

DEFAULT_PICKLE_LOCATION = '../data/pickled/'
DEFAULT_DATA_DIRECTORY = '../data/aclImdb/'
PRINT_DEBUG = True
NUM_WORDS = Settings.NUM_WORDS
NUM_CLASSES = Settings.NUM_CLASSES

W2V_M1_LOC = "../data/word2vec_models/w2v_m1.model"
W2V_M2_LOC = "../data/word2vec_models/w2v_m2.model"
W2V_M3_LOC = "../data/word2vec_models/w2v_m3.model"

VAL_SIZE = Settings.VAL_SIZE

ITEMS_TO_REMOVE = ['<br />', '.', ',', '*', '-', '%', '$', ';', '=', '[', ']',
        '(', ')', '&', '#', '@']

def gen_image_format(review, wv_m1, wv_m2, wv_m3):
    emb_review = []
    vocab = wv_m1.vocab.keys()
    for wv in [wv_m1, wv_m2, wv_m3]:
        item = \
            [wv.vocab[word.lower()].index for word in review.split() if word in vocab]
# FIXME: Decide what should be filled with. At the moment, its filled with 0
# index which might map to some word. Figure out what we should map to instead.
        while len(item) < NUM_WORDS:
#            emb_review.append(wv.vocab[" "].index)
            item.append(0)
        emb_review.append(np.array(item[:NUM_WORDS]).flatten())

    return np.array(emb_review)

def convert_rating(rating):
    ''' Helper function to convert 'rating' column of the dataframes to 1 of 4 
    values
    '''
    if rating < 3:
        return 0
    elif rating < 5:
        return 1 if NUM_CLASSES == 4 else 0
    elif rating < 9:
        return 2 if NUM_CLASSES == 4 else 1
    else:
        return 3 if NUM_CLASSES == 4 else 1

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

# TODO: Need to filter the data such that it 299 in 

    # Don't have a pickled file. Make it
    data = get_IMDB(train=train, positive=positive)
    data['embedding'] = [gen_image_format(sub, wv_m1, wv_m2, wv_m3) for sub in
        data['review'].values]
    data.to_pickle(file_dir)
    return data

def dataset_IMDB(train=True, val=False):
    # Get the positive and negative datasets
    pos = get_emb_IMDB(train=train, positive=True)
    neg = get_emb_IMDB(train=train, positive=False)

    
    size = int(VAL_SIZE/2)
    if val == True:
        data = pd.concat([pos[:size], neg[:size]])
        return data.sample(frac=1)
    elif train == False:
        data = rd.concat([pos[:size], neg[:size]])
        return data.sample(frac=1)
    else:
        # combine the dataset, shuffle the data.
        data = pd.concat([pos, neg])
        return data.sample(frac=1)
    

if __name__ == '__main__':
    # Build all the normal dataframes.
    print("Train, pos:", get_IMDB(train=True, positive=True).shape)
    print("Train neg:", get_IMDB(train=True, positive=False).shape)
    print("Test pos:", get_IMDB(train=False, positive=True).shape)
    print("Test neg:", get_IMDB(train=False, positive=False).shape)



