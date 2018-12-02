# This is my playground for testing.


#import datetime
#import time
#import logging
#logging.basicConfig(filename='example.log',level=logging.DEBUG)
#
#logging.info(datetime.datetime.now().strftime("%d-%m-%Y %I:%M:%S %p"))
#logging.debug('This message should go to the log file')
#logging.info('So should this')
#logging.warning('And this, too')
#

import datetime
import logging
import numpy as np
import os 
import pandas as pd
import sys
import tensorflow as tf

sys.path.insert(0, '../helpers/') 

import LoadIMDB
import LoadEmbeddings
import MyInception


cpu = 'cpu:0'
if USE_GPU:
    device = '/device:GPU:0'
else:
    device = '/cpu:0' 

training = LoadIMDB.dataset_IMDB(train=True)
validation = LoadIMDB.dataset_IMDB(train=True, val=True)

x_training = training['embedding'].values
y_training = training['rating'].values

wv_m1 = LoadEmbeddings.get_wordvec(W2V_M1_LOC)
wv_m2 = LoadEmbeddings.get_wordvec(W2V_M2_LOC)
wv_m3 = LoadEmbeddings.get_wordvec(W2V_M3_LOC)

# Dimension reduce the array.
#temp_training = [x for x in x_training]
#x_training = np.array(temp_training)

x_training = np.array([x for x in x_training]) # unpack dataset.
validation = np.array([x for x in validation])

#    print("y_training", type(y_training), y_training.shape)
#    print(y_training[:10])

#    print(type(x_training), x_training.shape)
#    print(type(x_training[0]), x_training[0].shape)
#    print(type(x_training[0][0]), x_training[0][0].shape)
#    print(type(x_training[0][0][0]))


# Start building data tensors

# Weights must be on cpu because embedding doesn't support GPU
with tf.device(cpu):
    # Features used to train. Contains 3 channels. Must convert all 3
    inputs = tf.placeholder(tf.int32, [None], name='word_ids')

    x_train = tf.placeholder(tf.int32, shape=[None, 3, 299], name="raw_input")
    y_train = tf.placeholder(tf.int32, shape=[None,], name="rating")

    # Build a dataset #TODO: Adjust the batch size and maybe add shuffling
    train_dset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_it = train_dset.make_initializable_iterator()
    train_x_el, train_y_el = train_it.get_next()

    # Load the emebedding the cpu
    W1 = tf.constant(wv_m1.syn0, name="W1")
    W2 = tf.constant(wv_m1.syn0, name="W2")
    W3 = tf.constant(wv_m1.syn0, name="W3")

    # Tensor that will perform the conversion to true embedding vector. 
    # inputs should be of shape (299,)
    embedded1 = tf.nn.embedding_lookup(W1, inputs)
    embedded2 = tf.nn.embedding_lookup(W2, inputs)
    embedded3 = tf.nn.embedding_lookup(W3, inputs)

    # TODO: Check if this is valid.
#        built_sam = tf.Variable([embedded1, embedded2, embedded3])

# Main training enviornment.
with tf.device(device):
    x = tf.placeholder(tf.float32, [None, NUM_WORDS, EMB_DIM, 3],
        name="all_sam")
    y = tf.placeholder(tf.int32, [None, ])

#        is_training = tf.placeholder(tf.bool, name='is_training')

    # Build the model.
    scores = model_init_fn(x)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
            logits=scores)
    loss = tf.reduce_mean(loss)

    optimizer = optimizer_init_fn()
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss)


