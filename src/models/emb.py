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
import Settings


# Global Variables

USE_GPU = Settings.USE_GPU

LOG_TO_FILE = Settings.LOG_TO_FILE

LOG_FILENAME = "../data/logs/inception.log"
SAVE_MODEL = "../data/checkpoint/cp.ckpt" # Default save path of model.
SAVE = Settings.SAVE
LOAD_SAVED = Settings.LOAD_SAVED

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
#NUM_EPOCHS = 1000

def model_init_fn(inputs):
    ''' How do you want the model to instantiated '''
    return MyInception.MyInception(hidden_size=HID_SIZE, 
            num_fc_layers=NUM_LAYERS, num_classes=NUM_CLASSES)(inputs)

def optimizer_init_fn(learning_rate=LEARNING_RATE):
    ''' What type of optimizer do we want to use? '''
#    return tf.train.GradientDescentOptimizer(learning_rate)
    return tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
#    return tf.train.AdamOptimizer(learning_rate=learning_rate)

def check_accuracy(sess, val_dset, x, scores):
    """
    Check accuracy on a classification model.
    
    Inputs:
    - sess: A TensorFlow Session that will be used to run the graph
    - val_dset: A numpy array object on which to check accuracy
    - x: A TensorFlow placeholder Tensor where input images should be fed
    - scores: A TensorFlow Tensor representing the scores output from the
      model; this is the Tensor we will ask TensorFlow to evaluate.
      
    Returns: Nothing, but prints the accuracy of the model to the logger.
    """
    num_correct, num_samples = 0, 0
    x_batch = np.array([x for x,y in val_dset])
    y_batch = np.array([y for x,y in val_dset])
    feed_dict = {x: x_batch}
    scores_np = sess.run(scores, feed_dict=feed_dict)
    y_pred = scores_np.argmax(axis=1)
    num_samples += x_batch.shape[0]
    num_correct += (y_pred == y_batch).sum()
    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f%%)' % (num_correct, num_samples, 100*acc))
    return acc

def train(model_init_fn, optimizer_init_fn, num_epochs=1):
    ''' Test training method for converting embeddings before feeding into
    model
    '''
    cpu = 'cpu:0'
    if USE_GPU:
        device = '/device:GPU:0'
    else:
        device = '/cpu:0' 

    training = LoadIMDB.dataset_IMDB(train=True) # Load in the data we need.
    validation = LoadIMDB.dataset_IMDB(train=False, val=True)

    x_training = training['embedding'].values
    y_training = training['rating'].values
    del training

    x_val = validation['embedding'].values
    y_val = validation['rating'].values
    del validation

    wv_m1 = LoadEmbeddings.get_wordvec(W2V_M1_LOC)
    wv_m2 = LoadEmbeddings.get_wordvec(W2V_M2_LOC)
    wv_m3 = LoadEmbeddings.get_wordvec(W2V_M3_LOC)

    # Dimension reduce the array.
    x_training = np.array([x for x in x_training])
    x_val = np.array([x for x in x_val])

#    print(type(x_training), x_training.shape)
#    print(type(x_training[0]), x_training[0].shape)
#    print(type(x_training[0][0]), x_training[0][0].shape)
#    print(type(x_training[0][0][0]))

    # Start building data tensors

    # Weights must be on cpu because embedding doesn't support GPU
    with tf.device(cpu):
        # Features used to train. Contains 3 channels. Must convert all 3
        emb_input = tf.placeholder(tf.int32, [None,], name='word_ids')

        x_train = tf.placeholder(tf.int32, shape=[None, 3, NUM_WORDS], 
                name="raw_input")
        y_train = tf.placeholder(tf.int32, shape=[None,], name="rating")

        # Build train dataset #TODO: Adjust batch size and maybe add shuffling
        train_dset = tf.data.Dataset.from_tensor_slices(
                (x_train, y_train)).batch(BATCH_SIZE)
        train_it = train_dset.make_initializable_iterator()
        x_train_el, y_train_el = train_it.get_next()

        # Build validation dataset # TODO: Adjust batch size and add shuffling
        # Note we are reusing the x_training and y_training tensors.
        val_dset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        val_it = val_dset.make_initializable_iterator()
        x_val_el, y_val_el = val_it.get_next()

        # Load the emebedding the cpu
        W1 = tf.constant(wv_m1.syn0, name="W1")
        W2 = tf.constant(wv_m1.syn0, name="W2")
        W3 = tf.constant(wv_m1.syn0, name="W3")

        # Tensor that will perform the conversion to true embedding vector. 
        # inputs should be of shape (NUM_WORDS,)
        embedded1 = tf.nn.embedding_lookup(W1, emb_input)
        embedded2 = tf.nn.embedding_lookup(W2, emb_input)
        embedded3 = tf.nn.embedding_lookup(W3, emb_input)

        # TODO: Check if this is valid.
#        built_sam = tf.Variable([embedded1, embedded2, embedded3])

    # Main training enviornment.
    with tf.device(device):
        # These are direct inputs into the training model.
        x = tf.placeholder(tf.float32, [None, NUM_WORDS, EMB_DIM, 3],
            name="all_sam")
        y = tf.placeholder(tf.int32, [None, ])

#        is_training = tf.placeholder(tf.bool, name='is_training')

        # Build the model.
        scores = model_init_fn(x)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                logits=scores)

#        sm = tf.nn.softmax(scores)
#        loss = tf.losses.mean_squared_error(labels=y, predictions=scores)

        loss = tf.reduce_mean(loss)

        optimizer = optimizer_init_fn()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss)


    best_val_acc = 0.0
    saver = tf.train.Saver()
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        if (LOAD_SAVED and tf.train.checkpoint_exists(SAVE_MODEL)):
            saver.restore(sess, SAVE_MODEL)
#            logging.info("Restoring model at" + SAVE_MODEL)
        else:
            sess.run(tf.global_variables_initializer())
#            logging.info("Training fresh model")

        sess.run(train_it.initializer, feed_dict={x_train: x_training,
            y_train: y_training})

        # Pre embed the validiation dataset for quick val testing
        sess.run(val_it.initializer, feed_dict={x_train: x_val, y_train: y_val})

        my_validation = [] # This is the pre embedded validation set.
        print("Building validation set")
        while True:
            try:
                x_sam, y_sam = sess.run([x_val_el, y_val_el])
                channel1 = sess.run(embedded1, feed_dict={emb_input:x_sam[0]})
                channel2 = sess.run(embedded2, feed_dict={emb_input:x_sam[1]})
                channel3 = sess.run(embedded3, feed_dict={emb_input:x_sam[2]})

                word_image = np.array([channel1, channel2, channel3])
                word_image = np.reshape(word_image, [NUM_WORDS, EMB_DIM, 3])
                my_validation.append((word_image, y_sam))
            except tf.errors.OutOfRangeError:
                break
        my_validation = np.array(my_validation)

        t = 0
#        for epoch in range(num_epochs):
#
#            print('Starting epoch %d' % epoch)
#            while True:
#                try:
#                    # TODO:Scale this so that it can take more than just one
#                    # sample
#                    # Build all the channels for the data.
#                    x_sam, y_sam = sess.run([x_train_el, y_train_el])
#                    x_input = []
#                    for xx in x_sam:
#                        channel1 = sess.run(embedded1, feed_dict={emb_input:xx[0]})
#                        channel2 = sess.run(embedded2, feed_dict={emb_input:xx[1]})
#                        channel3 = sess.run(embedded3, feed_dict={emb_input:xx[2]})
#
#                        # Word encoding in the shape of an image
#                        word_image = np.array([channel1, channel2, channel3])
#                        word_image = np.reshape(word_image, [NUM_WORDS, EMB_DIM, 3])
#                        x_input.append(word_image)
#
#                    feed_dict = {x: np.array(x_input), y: np.array(y_sam)}
#                    loss_np, _ = sess.run([loss, train_op], feed_dict=feed_dict)
#
#
#                    if t % PRINT_EVERY == 0:
#                        print('Iteration %d, loss = %.4f' % (t, loss_np))
##                        logging.info('Iteration %d, loss = %.4f' % (t, loss_np))
##                        print("start check acc")
#                        acc = check_accuracy(sess, my_validation, x, scores)
#                        if SAVE and acc < best_val_acc:
#                            save_path = saver.save(sess, SAVE_MODEL)
#                            best_loss = loss_np
#                            print("Iteration %d, loss = %.4f, (val_acc = %.2): \ Model saved in path %s" % (t, loss_np, save_path, 100*acc))
#
#                    t += 1
#                except tf.errors.OutOfRangeError:
##                    logging.info("Complete epoch: " + (epoch + 1))
#                    pass

    
if __name__ == "__main__":

    # TODO: CHANGE DEBUG WARNING TO INFO
    log_level = logging.WARNING

    if LOG_TO_FILE:
        logging.basicConfig(
                filename=LOG_FILENAME,
                format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s] %(message)s",
                level=log_level)
    else:
        logging.basicConfig(
                format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s] %(message)s",
                level=log_level)

    logging.info("Train model")
    train(model_init_fn, optimizer_init_fn)
