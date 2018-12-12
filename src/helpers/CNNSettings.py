# Contains all the parameters for model training.

USE_TRANSFER = False # LEAVE FALSE!

####### Training params ########
SAVE= False # Do we save the model?
LOAD_SAVED = True # Do load a model?
USE_GPU = True 
LOG_TO_FILE = True


VAL_SIZE = 500
BATCH_SIZE = 10
PRINT_EVERY = 200

NUM_CLASSES = 2

####### Model params ########
EMB_DIM = 75
NUM_WORDS = 75
NUM_CHANNELS = 3

MOMENTUM = 0.9
LEARNING_RATE = 1e-2
DROPOUT_RATE = 0.3
HID_SIZE = 128

NUM_LAYERS = 3

####
CHANNEL_1 = 13
CHANNEL_2 = 5

MAX_COLOR = 255
