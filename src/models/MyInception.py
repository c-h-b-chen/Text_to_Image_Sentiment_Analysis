import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
from tensorflow.keras.applications import InceptionV3

NUM_CLASSES = 2
NUM_HID_LAYERS = 64
NUM_FC_LAYERS = 5

class MyInception(tf.keras.Model):

    def __init__(self, num_classes=NUM_CLASSES, hidden_size=NUM_HID_LAYERS, 
            num_fc_layers=NUM_FC_LAYERS):
        super().__init__()

        initializer = tf.variance_scaling_initializer(scale=2.0)

        # FIXME: Include an input_shape
        # pooling so that we have an output 2D tensor. to work with. Make sure
        # to flatten it.
        self.cnn = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
        # Make sure to not adjust the weights of the transfer model.
        cnn.trainable = False 

        # TODO: Add the word2vec layers here.

        self.fullyConnected = tf.keras.Sequential()
        # Build the fully connect layers.
        for _ in range(num_fc_layers):
            self.fullyConnected.add(tf.keras.layers.Dense(64, activation='relu',
                kernel_initializer=initializer))
        self.output_layer = tf.keras.layers.Dense(4)
        self.sm = tf.keras.Softmax()

    # Forward pass through the training loop.
    def call(self, x, training=None):
        pass


# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Dense(64, activation='relu'))
# model.add(tf.keras.layers.Dense(64, activation='relu'))
# model.add(tf.keras.layers.Dense(10, activation='softmax'))
# 
# # Check if using GPUs
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
# 
# video = keras.Input(shape=(None, 150, 150, 3), name='video')
# cnn = InceptionV3(weights='imagenet',
#                   include_top=False,
#                   pooling='avg')
# 
# cnn.trainable = False
# frame_features = layers.TimeDistrivuted(cnn)(video)
# video_vector = layers.LSTM(256)(frame_features)
#
# # Turn a sequence of words into a vector
# review = keras.Input(shape=(None,), dtype='int32', name='review')
# embedded_words = layers.Embedding(input_voc_size, 256)(review)
# question_vector = layers.LSTM(128)(embedded_words)

