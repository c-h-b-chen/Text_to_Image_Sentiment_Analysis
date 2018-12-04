import sys
sys.path.insert(0, '../helpers/') 
import tensorflow as tf

import Settings
# from tensorflow import keras
# from tensorflow.keras import layers
from tensorflow.keras.applications import InceptionV3

NUM_CLASSES = Settings.NUM_CLASSES
NUM_HID_STATES = Settings.HID_SIZE
NUM_FC_LAYERS = 2 if Settings.NUM_LAYERS < 2 else Settings.NUM_LAYERS

EMB_DIM = Settings.EMB_DIM

class MyCNN(tf.keras.Model):
    def __init__(self, channel1, channel2, hidden_size=NUM_HID_STATES,
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

#        self.cnn1 = tf.layers.Conv2D(channel1, [EMB_DIM,5], 1,'same',
        self.cnn1 = tf.layers.Conv2D(channel1, [5,5], 1,'same',
                activation=tf.nn.relu, kernel_initializer=initializer)

#        self.cnn2 = tf.layers.Conv2D(channel2, [EMB_DIM,3], 1,'same',
        self.cnn2 = tf.layers.Conv2D(channel2, [3,3], 1,'same',
                activation=tf.nn.relu, kernel_initializer=initializer)

        self.fullyConnected = tf.keras.Sequential()

        # Build the fully connect layers.
        for _ in range(num_fc_layers):
            self.fullyConnected.add(tf.keras.layers.Dense(NUM_HID_STATES, 
                activation='relu', kernel_initializer=initializer))

        self.output_layer = tf.keras.layers.Dense(num_classes,
                kernel_initializer=initializer)

    # Forward pass through the training loop when we call the model.
    def call(self, x, training=None):
        ''' Expects input of the form (NUM_WORDS, EMB_DIM, 3). Returns 4 values
        for each classification method '''

        # On input data, we will need to convert the embed vectors into their
        # true embeddings using out embedding models.
        # embedded = tf.nn.embedding_lookup(embeddings, inputs)
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = tf.layers.flatten(x)
        x = self.fullyConnected(x)
        scores = self.output_layer(x)
        return scores


def test_MyCNN():
    device = "cpu:0" # Test on CPU for now.

    tf.reset_default_graph()
    hidden_size, num_fc_layers, num_classes = 64, 2, 4
    model = MyCNN(13, 5, hidden_size=hidden_size, num_fc_layers=num_fc_layers,
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
