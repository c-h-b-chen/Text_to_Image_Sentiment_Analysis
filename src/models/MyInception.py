import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import InceptionV3

class MyInception(tf.keras.Model):
    def __init__(self, hidden_size, num_classes):
        super().__init__()
        initializer = tf.variance_scaling_initializer(scale=2.0)


    # Forward pass through the training loop.
    def call(self, x, training=None):


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

