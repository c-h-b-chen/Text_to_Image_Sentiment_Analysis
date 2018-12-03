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

# Default gensim wordvec models to use
W2V_M1_LOC = "../data/word2vec_models/w2v_m1.model"
W2V_M2_LOC = "../data/word2vec_models/w2v_m2.model"
W2V_M3_LOC = "../data/word2vec_models/w2v_m3.model"

NUM_WORDS = 299
EMB_DIM = 299
NUM_CHANNELS = 3
HID_SIZE = 64
NUM_LAYERS = 3
NUM_CLASSES = 4
PRINT_EVERY = 100


USE_GPU = False
cpu = 'cpu:0'
if USE_GPU:
    device = '/device:GPU:0'
else:
    device = '/cpu:0' 

training = LoadIMDB.dataset_IMDB(train=True)
#validation = LoadIMDB.dataset_IMDB(train=True, val=True)

x_training = training['embedding'].values
y_training = training['rating'].values

wv_m1 = LoadEmbeddings.get_wordvec(W2V_M1_LOC)
wv_m2 = LoadEmbeddings.get_wordvec(W2V_M2_LOC)
wv_m3 = LoadEmbeddings.get_wordvec(W2V_M3_LOC)

# Dimension reduce the array.
#temp_training = [x for x in x_training]
#x_training = np.array(temp_training)

x_training = np.array([x for x in x_training]) # unpack dataset.
#validation = np.array([x for x in validation])

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
    x_train_el, y_train_el = train_it.get_next()

    # Load the emebedding the cpu
    W1 = tf.constant(wv_m1.syn0, name="W1")
    W2 = tf.constant(wv_m1.syn0, name="W2")
    W3 = tf.constant(wv_m1.syn0, name="W3")

    # Tensor that will perform the conversion to true embedding vector. 
    # inputs should be of shape (299,)
    embedded1 = tf.nn.embedding_lookup(W1, inputs)
    embedded2 = tf.nn.embedding_lookup(W2, inputs)
    embedded3 = tf.nn.embedding_lookup(W3, inputs)

t=0
my_training = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(train_it.initializer, feed_dict={x_train: x_training,
        y_train: y_training})
    print("start converting")
    while True:
        try:
            print(t),
            t = t + 1
            x_sam, y_sam = sess.run([x_train_el, y_train_el])
            channel1 = sess.run(embedded1, feed_dict={inputs:x_sam[0]})
            channel2 = sess.run(embedded1, feed_dict={inputs:x_sam[1]})
            channel3 = sess.run(embedded1, feed_dict={inputs:x_sam[2]})

            word_image = np.array([channel1, channel2, channel3])
            word_image = np.reshape(word_image, [299, 299, 3])
            my_training.append((word_image, y_train_el))
        except tf.errors.OutOfRangeError:
            print("done")
    np.save("deleteMe.npy", my_training)











## Main training enviornment.
#with tf.device(device):
#    x = tf.placeholder(tf.float32, [None, NUM_WORDS, EMB_DIM, 3],
#        name="all_sam")
#    y = tf.placeholder(tf.int32, [None, ])
#
##        is_training = tf.placeholder(tf.bool, name='is_training')
#
#    # Build the model.
#    scores = model_init_fn(x)
#
#    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
#            logits=scores)
#    loss = tf.reduce_mean(loss)
#
#    optimizer = optimizer_init_fn()
#    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#    with tf.control_dependencies(update_ops):
#        train_op = optimizer.minimize(loss)
#
#
