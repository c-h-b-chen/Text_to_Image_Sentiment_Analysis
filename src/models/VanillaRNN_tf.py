# Vanilla RNN used for semantic analysis build with low-level tensorflow.

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

class RNN_Model(object):
    '''
    Description:
    Function:
        __init__() -- Build the RNN object instantiate all basic parameters.
            Loads any parameters that might be saved. 
        predict() -- Generate labels based on input.
        score() -- Generate labels and determine the accuaracy of the model
            based on the inputs '''

    def __init__(self, num_layers=10, training=True):
        '''
        Build the main functionality of the model

        Notes: Delete this
          1) Need a way to load in past parameters to allow for more training
          if needed
          2) Build the main data input stream into the model.
          3) Build the each node.
          4) Define the execution graph that will be ran when called on.
        '''
        pass

    def train(lr=0.001):
        pass

    def predict():
        pass

    def score():
        pass

if __name__ == "__main__":
    print("Checking model compilation")
    test_model = RNN_Model()
    print("Good to go")

# Note: Should compate with the keras RNN to test performance of
#   implementation.
