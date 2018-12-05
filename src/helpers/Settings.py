# Contains all the parameters for model training.

USE_TRANSFER = True # When false, train a 2 layer cnn instead of the transfer

####### Training params ########
SAVE= True # Do we save the model?
LOAD_SAVED = True # Do load a model?
USE_GPU = True 
LOG_TO_FILE = True


VAL_SIZE = 500
BATCH_SIZE = 30
PRINT_EVERY = 10

NUM_CLASSES = 4

####### Model params ########
EMB_DIM = 75
NUM_WORDS = 75
NUM_CHANNELS = 3

MOMENTUM = 0.9
LEARNING_RATE = 0.01
DROPOUT_RATE = 0.3
HID_SIZE = 128

NUM_LAYERS = 3

####
CHANNEL_1 = 13
CHANNEL_2 = 5
