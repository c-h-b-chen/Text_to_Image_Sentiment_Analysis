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


# Global Variables

USE_GPU = True

LOG_TO_FILE = False

LOG_FILENAME = "../data/logs/inception.log"
SAVE_MODEL = "../data/checkpoint/cp.ckpt" # Default save path of model.

# Default gensim wordvec models to use
W2V_M1_LOC = "../data/word2vec_models/w2v_m1.model"
W2V_M2_LOC = "../data/word2vec_models/w2v_m2.model"
W2V_M3_LOC = "../data/word2vec_models/w2v_m3.model"
LOG_FORMAT="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s] %(message)s",

EMB_DIM = 299
NUM_CHANNELS = 3
NUM_WORDS = 299
HID_SIZE = 64

NUM_CLASSES = 4
NUM_LAYERS = 3

BATCH_SIZE = 50
PRINT_EVERY = 5
#NUM_EPOCHS = 1000



def check_accuracy(sess, x_val_el, y_val_el, x, scores):
    """
    Check accuracy on a classification model.
    
    Inputs:
    - sess: A TensorFlow Session that will be used to run the graph
    - x_val_el: Tensor to get the next iterator x_val from the iterator.
    - y_val_el: Tensor to get the next iterator y_val from the iterator.
    - x: A TensorFlow placeholder Tensor where input images should be fed
    - scores: A TensorFlow Tensor representing the scores output from the
      model; this is the Tensor we will ask TensorFlow to evaluate.
      
    Returns: Nothing, but prints the accuracy of the model to the logger.
    """
    num_correct, num_samples = 0, 0

    while True:
        try:
#            x_batch, y_batch = sess.run([x_val_el
            for x_batch, y_batch in dset:
                feed_dict = {x: x_batch}
                scores_np = sess.run(scores, feed_dict=feed_dict)
                y_pred = scores_np.argmax(axis=1)
                num_samples += x_batch.shape[0]
                num_correct += (y_pred == y_batch).sum()
        except tf.errors.OutOfRangeError:
            acc = float(num_correct) / num_samples
            print('Got %d / %d correct (%.2f%%)' % (num_correct, num_samples, 100*acc))


def model_init_fn(inputs):
    ''' How do you want the model to instantiated '''
    return MyInception.MyInception(hidden_size=HID_SIZE, 
            num_fc_layers=NUM_LAYERS, num_classes=NUM_CLASSES)(inputs)

def optimizer_init_fn():
    ''' What type of optimizer do we want to use? '''
    learning_rate = 0.01
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

    training = LoadIMDB.dataset_IMDB(train=True) # Load in the data we need.
    validation = LoadIMDB.dataset_IMDB(train=False, val=True)

    x_training = training['embedding'].values
    y_training = training['rating'].values

    wv_m1 = LoadEmbeddings.get_wordvec(W2V_M1_LOC)
    wv_m2 = LoadEmbeddings.get_wordvec(W2V_M2_LOC)
    wv_m3 = LoadEmbeddings.get_wordvec(W2V_M3_LOC)

    # Dimension reduce the array.
#    temp_training = [x for x in x_training]
#    x_training = np.array(temp_training)
    x_training = np.array([x for x in x_training])
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
        emb_input = tf.placeholder(tf.int32, [None,], name='word_ids')

        x_train = tf.placeholder(tf.int32, shape=[None, 3, 299], 
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
        # inputs should be of shape (299,)
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
        loss = tf.reduce_mean(loss)

        optimizer = optimizer_init_fn()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss)


    best_loss = 10000 # FIXME: Change this arbitrary loss value.
    saver = tf.train.Saver()
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        if (tf.train.checkpoint_exists(SAVE_MODEL)):
            saver.restore(sess, SAVE_MODEL)
#            logging.info("Restoring model at", SAVE_MODEL)
        else:
            sess.run(tf.global_variables_initializer())
#            logging.info("Training fresh model")

        sess.run(train_it.initializer, feed_dict={x_train: x_training,
            y_train: y_training})

        t = 0
        for epoch in range(num_epochs):
            print('Starting epoch %d' % epoch)
            while True:
                try:
                    # TODO:Scale this so that it can take more than just one
                    # sample
                    # Build all the channels for the data.
                    x_sam, y_sam = sess.run([x_train_el, y_train_el])
#                    print("x_sam.shape", x_sam.shape)
#                    print("y_sam.shape", y_sam.shape)
                    x_input = []
                    for xx in x_sam:
                        channel1 = sess.run(embedded1, feed_dict={emb_input:xx[0]})
                        channel2 = sess.run(embedded2, feed_dict={emb_input:xx[1]})
                        channel3 = sess.run(embedded3, feed_dict={emb_input:xx[2]})

                        # Word encoding in the shape of an image
                        word_image = np.array([channel1, channel2, channel3])
                        word_image = np.reshape(word_image, [NUM_WORDS, EMB_DIM, 3])
                        x_input.append(word_image)

#                    print("word_img shape", word_image.shape)
#                    print("y_sample", type(y_sample), y_sample)

                    feed_dict = {x: np.array(x_input), y: np.array(y_sam)}
                    loss_np, _ = sess.run([loss, train_op], feed_dict=feed_dict)

                    # TODO: Need to use validation accuracy instead of loss.
                    if loss_np < best_loss:
                        save_path = saver.save(sess, SAVE_MODEL)
                        best_loss = loss_np
                        print("Iteration", t+1, "Loss", loss_np, 
                                "Model saved in path: %s" % save_path)
#                        logging.info("Iterations", t, "Loss",
#                                loss_np, "Model saved in path: %s" % save_path)

                    if t % PRINT_EVERY == 0:
                        print('Iteration %d, loss = %.4f' % (t, loss_np))
                        logging.info('Iteration %d, loss = %.4f' %
                                (t, loss_np))
                        check_accuracy(sess, val_dset, x, scores)
                    t += 1
                except tf.errors.OutOfRangeError:
                    logging.info("Complete epoch:", epoch + 1)
                    pass

    
if __name__ == "__main__":

    # TODO: CHANGE DEBUG WARNING TO INFO
    log_level = logging.WARNING

    if LOG_TO_FILE:
        logging.basicConfig( filename=LOG_FILENAME, format=LOG_FORMAT,
                level=log_level)
    else:
        logging.basicConfig( format=LOG_FORMAT, level=log_level)

    logging.info("Train model")
    train(model_init_fn, optimizer_init_fn)
