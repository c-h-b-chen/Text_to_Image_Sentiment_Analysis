# Vanilla RNN used for semantic analysis build with keras tensorflow.
import tensorflow as tf
from tensorflow.keras import layers

class RNN_Model(tf.keras.Model):
    def __init__(self, num_classes=2):
        super(RNN_Model, self).__init__(name="RNN_Model")
        self.num_classes = num_classes
        # TODO: Figure out how to instantiate the RNN cells.
        self.rnn_1 = layers.LSTMCell(128)
        # self.rnn_1 = layers.CuDNNLSTM(128) # GPU accelerated cell

    # Forward pass through the model.
    def call(self, inputs):
        pass

if __name__ == "__main__":
    print("Checking model compilation")
    test_model = RNN_Model()
    print("Good to go")

# Note: Should compate with the keras RNN to test performance of
#   implementation.
