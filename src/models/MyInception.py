import sys
sys.path.insert(0, '../helpers/') 
import tensorflow as tf

import Settings
# from tensorflow import keras
# from tensorflow.keras import layers
from tensorflow.keras.applications import InceptionV3

NUM_CLASSES = Settings.NUM_CLASSES
NUM_HID_STATES = Settings.HID_SIZE
NUM_FC_LAYERS = Settings.NUM_LAYERS

class MyInception(tf.keras.Model):
#    def __init__(self, wv1, wv2, wv3, hidden_size=NUM_HID_LAYERS,
    def __init__(self, hidden_size=NUM_HID_STATES,
            num_fc_layers=NUM_FC_LAYERS, num_classes=NUM_CLASSES):
        ''' Constructor of our Imagenet/Inception Model for sentiment analysis.
        params:
            hidden_size -- Number of hidden states in FC layers
            num_fc_layers -- Number of fully connected layers at end of model.
            num_classes -- How many classes are there to predict.
        '''
        super().__init__()

        # Default initializer we will use.
        initializer = tf.variance_scaling_initializer(scale=2.0)

        # Pooling so that we have an output 2D tensor. to work with. Make sure
        # to flatten it.
        self.cnn = InceptionV3(weights='imagenet', include_top=False,
                pooling='avg')
        self.cnn.trainable = False 

        self.fullyConnected = tf.keras.Sequential()

        # Build the fully connect layers.
        for _ in range(num_fc_layers):
            self.fullyConnected.add(tf.keras.layers.Dense(NUM_HID_STATES, 
                activation='relu', kernel_initializer=initializer))

        self.output_layer = tf.keras.layers.Dense(num_classes,
                kernel_initializer=initializer)

    # Forward pass through the training loop when we call the model.
    def call(self, x, training=None):
        ''' Expects input of the form (299, 299, 3). Returns 4 values for each
        classification method '''

        # On input data, we will need to convert the embed vectors into their
        # true embeddings using out embedding models.
        # embedded = tf.nn.embedding_lookup(embeddings, inputs)
        x = self.cnn(x)
        x = tf.layers.flatten(x)
        x = self.fullyConnected(x)
        scores = self.output_layer(x)
        return scores


def test_MyInception():
    '''
    Tests basic functionality of MyInception. Used to make sure input pipeline
    flows as expected
    '''
    device = "cpu:0" # Test on CPU for now.

    tf.reset_default_graph()
    hidden_size, num_fc_layers, num_classes = 64, 2, 4
    model = MyInception(hidden_size=hidden_size, num_fc_layers=num_fc_layers,
            num_classes=num_classes)

    with tf.device(device):
        x = tf.zeros((64, 299, 299, 3)) # 64 samples, 3 channels, 299x299
        scores = model(x)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        scores_np = sess.run(scores)
        print(scores_np.shape) # Should be (64, 4)


if __name__ == "__main__":
    print("testing MyInception")
    test_MyInception()





















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

