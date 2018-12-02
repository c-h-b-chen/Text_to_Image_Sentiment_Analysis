import tensorflow as tf

import numpy as np
import pandas as pd
import os 
import sys
sys.path.insert(0, '../helpers/') 

# from LoadIMDB import dataset_IMDB
# from LoadEmbeddings import get_wordvec

import LoadIMDB
import LoadEmbeddings
import MyInception

#USE_GPU = False
USE_GPU = False

NUM_WORDS = 299
EMB_DIM = 299
NUM_CHANNELS = 3
HID_SIZE = 64
NUM_LAYERS = 3
NUM_CLASSES = 4

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
    print('Got %d / %d correct (%.2f%%)' % (num_correct, num_samples, 100*acc))

def model_init_fn(inputs):
    ''' How do you want the model to instantiated '''
    return MyInception.MyInception(hidden_size=HID_SIZE, num_fc_layers=NUM_LAYERS,
            num_classes=NUM_CLASSES)(inputs)

def optimizer_init_fn():
    ''' What type of optimizer do we want to use? '''
    learning_rate = 0.1
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

    training = LoadIMDB.dataset_IMDB(train=True)

    x_training = training['embedding'].values
    y_training = training['rating'].values

    wv_m1 = LoadEmbeddings.get_wordvec(W2V_M1_LOC)
    wv_m2 = LoadEmbeddings.get_wordvec(W2V_M2_LOC)
    wv_m3 = LoadEmbeddings.get_wordvec(W2V_M3_LOC)

    # Dimension reduce the array.
    temp_training = [x for x in x_training]
    x_training = np.array(temp_training)

    print(type(x_training), x_training.shape)
    print(type(x_training[0]), x_training[0].shape)
    print(type(x_training[0][0]), x_training[0][0].shape)
    print(type(x_training[0][0][0]))

    # Start building data tensors

    # Weights must be on cpu because embedding doesn't support GPU
    with tf.device(cpu):
        # Features used to train. Contains 3 channels. Must convert all 3
        inputs = tf.placeholder(tf.int32, [None], name='word_ids')

        x_train = tf.placeholder(tf.int32, shape=[None, 3, 299])
        y_train = tf.placeholder(tf.int32, shape=[None,])

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

    # Main training enviornment.
    with tf.device(device):
        x = tf.placeholder(tf.float32, [None, NUM_WORDS, EMB_DIM, 3])
        y = tf.placeholder(tf.int32, [None, ])

        # Build the model.
        scores = model_init_fn(x)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                logits=scores)
        loss = tf.reduce_mean(loss)

        optimizer = optimizer_init_fn()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(train_it.initializer, feed_dict={x_train: x_training,
            y_train: y_training})

        # Build all the channels for the data.
        x_sample, y_sample = sess.run([train_x_el, train_y_el])
        channel1 = sess.run(embedded1, feed_dict={inputs : x_sample[0]})
        channel2 = sess.run(embedded2, feed_dict={inputs : x_sample[1]})
        channel3 = sess.run(embedded3, feed_dict={inputs : x_sample[2]})

        # Word encoding in the shape of an image
        word_image = np.array([channel1, channel2, channel3])
        word_image = tf.reshape(word_image, [299, 299, 3])

#        print(word_image.shape)

        # Begin training the model now
        t = 0
        for epoch in range(num_epochs):
            print('Starting epoch %d' % epoch)
            while True:
            for x_np, y_np in sess.run(train_it.get_next()):
                feed_dict = {x: x_np, y: y_np, is_training:1}
                loss_np, _ = sess.run([loss, train_op], feed_dict=feed_dict)
                if t % print_every == 0:
                    print('Iteration %d, loss = %.4f' % (t, loss_np))
#                    check_accuracy(sess, val_dset, x, scores, is_training=is_training)
                    print()
                t += 1


## TODO: Refactor: Pretty large function.
## TODO: Allow for continue training of model
#def demo_train(model_init_fn, optimizer_init_fn, num_epcohs=1):
#    ''' This is the main training loop for the function'''
#    if USE_GPU:
#        device = '/device:GPU:0'
#    else:
#        device = '/cpu:0' 
#
#    cpu = '/cpu:0' # Embedding transformation only supports cpu usage.
#
#    # Load df train and test
#    training = LoadIMDB.dataset_IMDB(train=True)
#    test = LoadIMDB.dataset_IMDB(train=False)
#
#    # Load up the gensim w2v models. These models are not trainable.
#    wv_m1 = LoadEmbeddings.get_wordvec(W2V_M1_LOC)
#    wv_m2 = LoadEmbeddings.get_wordvec(W2V_M2_LOC)
#    wv_m3 = LoadEmbeddings.get_wordvec(W2V_M3_LOC)
#
#    #test = np.array(training['embedding'].values)
#    #np.array(training['embedding'].values)
#    #np.array(training['rating'].values)
#    
#    # Our input Tensors (training data)
#    x_train = tf.placeholder(tf.int32, shape=[None,3])
#    y_train = tf.placeholder(tf.int32, shape=[None,])
#
#    # Our input Tensors (testing data)
#    x_test = tf.placeholder(tf.int32, shape=[None,3])
#    y_test = tf.placeholder(tf.int32, shape=[None,])
#
#    # Build tensorflow datasets for efficient dataflow.
#    train_dset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
#    test_dset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
#
#    # Make data iterators
#    train_iter = train_dset.make_initializable_iterator()
#    test_iter = test_dset.make_initializable_iterator()
#
#    # Prime the iterators, this is the Tensor we want to call to get the data.
#    train_el = train_iter.get_next()
#    test_el = test_iter.get_next()
#
#    # Input data which contains word indices from embedding. Each sample will
#    # have 299 words, with 3 values for each of the word vector embeddings.
#    inputs = tf.placeholder(tf.int32, [NUM_WORDS,NUM_CHANNELS], name="word_ids")
#
#    # This is where the training embedded values from the WVs are. They are not
#    # trainable! This is the simplest approach although it does take up a lot
#    # of space.
#    # https://stackoverflow.com/questions/35687678/using-a-pre-trained-word-embedding-word2vec-or-glove-in-tensorflow
#    # TODO: Verify syn0 is the best embedding to use in this case.
#    # Other options are 'syn1' or 'syn1neg' : 
#    # https://stackoverflow.com/questions/41162876/get-weight-matrices-from-gensim-word2vec
#    emb1 = tf.constant(wv_m1.syn0, name="emb1")
#    emb2 = tf.constant(wv_m2.syn0, name="emb2")
#    emb3 = tf.constant(wv_m3.syn0, name="emb3")

    
if __name__ == "__main__":
    train(model_init_fn, optimizer_init_fn)
#    demo_train(model_init_fn, optimizer_init_fn)
