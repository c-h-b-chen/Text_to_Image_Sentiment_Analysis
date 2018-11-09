# Vanilla RNN used for semantic analysis build with low-level tensorflow.
# Note: Should compate with the keras RNN to test performance of
#   implementation.

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

class RNN_Model(object):
    '''
    Description:
    Function:
        __init__() -- Build the RNN object instantiate all basic parameters.
        train() -- Train the model
        predict()
        score()
    '''

    def __init__(self, num_layers=10, training=True):
        ''' Constructor '''
        # Is this model used for training. If so, then we need to update the
        # weights everytime we pass in inputs.
        self.training = training; 
        self.input_features = tf.placeholder(tf.float32, [None,])
        self.input_labels = tf.placeholder(tf.int32, [None,])
        cells = []
        for _ in range(num_layers):
            cell = rnn.RNNCell
            cell = rnn.DropoutWrapper(cell, input_keep_prob=True,
                output_keep_prob=True)
            cells.append(cell)
        self.cell = cell = rnn.MultiRNNCell(cells, state_is_tuple=True)

    def train(lr=0.001):
        '''
        Train the model with 
        Params:
            lr -- learning rate for the optimization function.
        '''
        pass

    def predict():
        '''
        Use trained model to make predictions on unseen data
        '''
        pass

    def score():
        '''
        Get the accuracy of the sample
        '''
        pass

if __name__ == "__main__":
    print("Checking model compilation")
    test_model = RNN_Model()
    print("Good to go")

