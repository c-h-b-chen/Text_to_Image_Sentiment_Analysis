# Contains all the parameters for model training.

####### Training params ########
SAVE= True
LOAD_SAVED = True
USE_GPU = True
LOG_TO_FILE = False

USE_TRANSFER = False # When false, train a 2 layer cnn instead of the transfer

VAL_SIZE = 500
BATCH_SIZE = 100
PRINT_EVERY = 10

NUM_CLASSES = 4

####### Model params ########
EMB_DIM = 75
NUM_WORDS = 75
NUM_CHANNELS = 3

MOMENTUM = 0.9
LEARNING_RATE = 0.001
HID_SIZE = 32

NUM_LAYERS = 2

####
CHANNEL_1 = 13
CHANNEL_2 = 5
