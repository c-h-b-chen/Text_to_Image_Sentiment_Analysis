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
import MyCNN
import Settings


# Global Variables

USE_GPU = Settings.USE_GPU

LOG_TO_FILE = Settings.LOG_TO_FILE

USE_TRANSFER = False

SAVE_MODEL = "../data/checkpoint/MySaved/Demo_Incep/Incep1.ckpt" if Settings.USE_TRANSFER else \
    "../data/checkpoint/MySaved/Demo_CNN/CNN.ckpt"

SAVE = False
LOAD_SAVED = True

# Default gensim wordvec models to use
W2V_M1_LOC = "../data/word2vec_models/w2v_m1.model"
W2V_M2_LOC = "../data/word2vec_models/w2v_m2.model"
W2V_M3_LOC = "../data/word2vec_models/w2v_m3.model"

EMB_DIM = Settings.EMB_DIM
NUM_CHANNELS = Settings.NUM_CHANNELS
NUM_WORDS = Settings.NUM_WORDS
LEARNING_RATE = Settings.LEARNING_RATE
MOMENTUM = Settings.MOMENTUM
HID_SIZE = Settings.HID_SIZE

NUM_CLASSES = Settings.NUM_CLASSES
NUM_LAYERS = Settings.NUM_LAYERS

BATCH_SIZE = Settings.BATCH_SIZE
PRINT_EVERY = Settings.PRINT_EVERY

CHANNEL_1 = Settings.CHANNEL_1
CHANNEL_2 = Settings.CHANNEL_2


def model_init_fn(inputs):
    ''' How do you want the model to instantiated '''
    if USE_TRANSFER:
        return MyInception.MyInception(hidden_size=HID_SIZE, 
                num_fc_layers=NUM_LAYERS, num_classes=NUM_CLASSES)(inputs)
    else:
        return MyCNN.MyCNN(CHANNEL_1, CHANNEL_2, hidden_size=HID_SIZE, 
                num_fc_layers=NUM_LAYERS, num_classes=NUM_CLASSES)(inputs)
        

def optimizer_init_fn(learning_rate=LEARNING_RATE):
    ''' What type of optimizer do we want to use? '''
    return tf.train.GradientDescentOptimizer(learning_rate)


def train(model_init_fn, optimizer_init_fn, num_epochs=1):
    ''' Test training method for converting embeddings before feeding into
    model
    '''
    cpu = 'cpu:0'
    if USE_GPU:
        device = '/device:GPU:0'
    else:
        device = '/cpu:0' 

#    training = LoadIMDB.dataset_IMDB(train=True) # Load in the data we need.
#    validation = LoadIMDB.dataset_IMDB(train=False, val=True)

#    x_training = training['embedding'].values
#    y_training = training['rating'].values
#    del training

#    x_val = validation['embedding'].values
#    y_val = validation['rating'].values
#    del validation

    wv_m1 = LoadEmbeddings.get_wordvec(W2V_M1_LOC)
    wv_m2 = LoadEmbeddings.get_wordvec(W2V_M2_LOC)
    wv_m3 = LoadEmbeddings.get_wordvec(W2V_M3_LOC)

    # Dimension reduce the array.
#    x_training = np.array([x for x in x_training])
#    x_val = np.array([x for x in x_val])

    # Start building data tensors

    # Weights must be on cpu because embedding doesn't support GPU
    with tf.device(cpu):
        # Features used to train. Contains 3 channels. Must convert all 3
        emb_input = tf.placeholder(tf.int32, [None,], name='word_ids')

#        x_train = tf.placeholder(tf.int32, shape=[None, 3, NUM_WORDS], 
#                name="raw_input")
#        y_train = tf.placeholder(tf.int32, shape=[None,], name="rating")

        # Build train dataset #TODO: Adjust batch size and maybe add shuffling
#        train_dset = tf.data.Dataset.from_tensor_slices(
#                (x_train, y_train)).batch(BATCH_SIZE).shuffle(len(x_training))
#        train_it = train_dset.make_initializable_iterator()
#        x_train_el, y_train_el = train_it.get_next()

        # Build validation dataset # TODO: Adjust batch size and add shuffling
        # Note we are reusing the x_training and y_training tensors.
#        val_dset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
#        val_it = val_dset.make_initializable_iterator()
#        x_val_el, y_val_el = val_it.get_next()

        # Load the emebedding the cpu
        W1 = tf.constant(wv_m1.syn0, name="W1")
        W2 = tf.constant(wv_m1.syn0, name="W2")
        W3 = tf.constant(wv_m1.syn0, name="W3")

        # Save the keys, make sure each item in a part of the keys.
        vocab = wv_m1.vocab.keys()

        # Tensor that will perform the conversion to true embedding vector. 
        # inputs should be of shape (NUM_WORDS,)
        embedded1 = tf.nn.embedding_lookup(W1, emb_input)
        embedded2 = tf.nn.embedding_lookup(W2, emb_input)
        embedded3 = tf.nn.embedding_lookup(W3, emb_input)


    # Main training enviornment can be GPU if have one.
    with tf.device(device):
        # These are direct inputs into the training model.
        x = tf.placeholder(tf.float32, [None, NUM_WORDS, EMB_DIM, 3],
            name="all_sam")
        y = tf.placeholder(tf.int32, [None, ])

        # Build the model.
        scores = model_init_fn(x)
        pred = tf.nn.softmax(scores)

#        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
#                logits=scores)

#        loss = tf.reduce_mean(loss)

#        optimizer = optimizer_init_fn()
#        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#        with tf.control_dependencies(update_ops):
#            train_op = optimizer.minimize(loss)

#    best_val_acc = 0.0
    saver = tf.train.Saver()
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        saver.restore(sess, SAVE_MODEL)

        while True:
            review = input(
            "\n".join(["Please give me a review about 75 words long\n",
            "(ie. I was a bit skeptical about the concept behind this show. What",
            "saves it from banality is just how creative and edgy each episode",
            "is. The viewer has NO idea what is going to happen next. There is",
            "no formula and the tension is often ratcheded up to excruciating",
            "levels. There are tons of laughs here and the back stories are",
            "woven in expertly. So many comedies are played very hammy with lots",
            "of stereotypes.)\n\n >> "]))
            if review == "":
                break

            # Filter the input review
            xx = LoadIMDB.gen_image_format(review, wv_m1, wv_m2, wv_m2)

            channel1 = sess.run(embedded1, feed_dict={emb_input:xx[0]})
            channel2 = sess.run(embedded2, feed_dict={emb_input:xx[1]})
            channel3 = sess.run(embedded3, feed_dict={emb_input:xx[2]})

            word_image = np.array([channel1, channel2, channel3])
            word_image = np.reshape(word_image, [NUM_WORDS, EMB_DIM, 3])

            feed_dict = {x: np.array([word_image])}
            raw_score = sess.run(pred, feed_dict=feed_dict)
            print(raw_score)

            rating = np.argmax(raw_score)
            if rating < 2:
                print(":(")
            else:
                print(":)")


if __name__ == "__main__":
    train(model_init_fn, optimizer_init_fn)

