import tensorflow as tf

import numpy as np
import pandas as pd
import os 
import sys
sys.path.insert(0, '../helpers/') 

from LoadIMDB import dataset_IMDB
from LoadEmbeddings import get_wordvec

USE_GPU = False
NUM_WORDS = 299
EMB_DIM = 299
NUM_CHANNELS = 3

# Default gensim wordvec models to use
W2V_M1_LOC = "../data/word2vec_models/w2v_m1.model"
W2V_M2_LOC = "../data/word2vec_models/w2v_m2.model"
W2V_M3_LOC = "../data/word2vec_models/w2v_m3.model"


def check_accuracy(sess, dset, x, scores, is_training=None):
    """
    Check accuracy on a classification model.
    
    Inputs:
    - sess: A TensorFlow Session that will be used to run the graph
    - dset: A Dataset object on which to check accuracy
    - x: A TensorFlow placeholder Tensor where input images should be fed
    - scores: A TensorFlow Tensor representing the scores output from the
      model; this is the Tensor we will ask TensorFlow to evaluate.
      
    Returns: Nothing, but prints the accuracy of the model
    """
    num_correct, num_samples = 0, 0
    for x_batch, y_batch in dset:
        feed_dict = {x: x_batch, is_training: 0}
        scores_np = sess.run(scores, feed_dict=feed_dict)
        y_pred = scores_np.argmax(axis=1)
        num_samples += x_batch.shape[0]
        num_correct += (y_pred == y_batch).sum()
    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f%%)' % (num_correct, num_samples, 100 * acc))

def model_init_fn():
    pass

def optimizer_init_fn():
    pass

def train():
    ''' This is the main training loop for the function'''
    if USE_GPU:
        device = '/device:GPU:0'
    else:
        device = '/cpu:0' 

    # Load df train and test
    training = dataset_IMDB(train=True)
    test = dataset_IMDB(train=False)

    # Load up the gensim w2v models. These models are not trainable.
    wv_m1 = get_wordvec(W2V_M1_LOC)
    wv_m2 = get_wordvec(W2V_M2_LOC)
    wv_m3 = get_wordvec(W2V_M3_LOC)

#test = np.array(training['embedding'].values)
#np.array(training['embedding'].values)
#np.array(training['rating'].values)

    # Our input Tensors (training data)
    x_train = tf.placeholder(tf.int32, shape=[None,3])
    y_train = tf.placeholder(tf.int32, shape=[None,])

    # Our input Tensors (testing data)
    x_test = tf.placeholder(tf.int32, shape=[None,3])
    y_test = tf.placeholder(tf.int32, shape=[None,])

    # Build tensorflow datasets for efficient dataflow.
    train_dset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    # Make data iterators
    train_iter = train_dset.make_initializable_iterator()
    test_iter = test_dset.make_initializable_iterator()

    # Prime the iterators, this is the Tensor we want to call to get the data.
    train_el = train_iter.get_next()
    test_el = test_iter.get_next()

    # Input data which contains word indices from embedding. Each sample will
    # have 299 words, with 3 values for each of the word vector embeddings.
    inputs = tf.placeholder(tf.int32, [NUM_WORDS,NUM_CHANNELS], name="word_ids")

    # This is where the training embedded values from the WVs are. They are not
    # trainable! This is the simplest approach although it does take up a lot
    # of space.
    # https://stackoverflow.com/questions/35687678/using-a-pre-trained-word-embedding-word2vec-or-glove-in-tensorflow
    # TODO: Verify syn0 is the best embedding to use in this case.
    # Other options are 'syn1' or 'syn1neg' : # https://stackoverflow.com/questions/41162876/get-weight-matrices-from-gensim-word2vec
    emb1 = tf.constant(wv_m1.syn0, name="emb1")
    emb2 = tf.constant(wv_m2.syn0, name="emb2")
    emb3 = tf.constant(wv_m3.syn0, name="emb3")

    
if __name__ == "__main__":
    train()
