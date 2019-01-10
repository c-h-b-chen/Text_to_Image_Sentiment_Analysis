# Contains all the parameters for model training.

USE_TRANSFER = False # When false, train a 2 layer cnn instead of the transfer

####### File/Save params ########
SAVE= True # Do we save the model?
LOAD_SAVED = False # Do load a model or start from scatch.
USE_GPU = True 
LOG_TO_FILE = True


####### Data params ########
NUM_EPOCHS = 400
VAL_SIZE = 1000
BATCH_SIZE = 20
PRINT_EVERY = 300

TRAIN_SET_SIZE = 24000

NUM_CLASSES = 2

####### Model params ########
EMB_DIM = 75
NUM_WORDS = 75
NUM_CHANNELS = 3

MOMENTUM = 0.9
LEARNING_RATE = 1e-3
DROPOUT_RATE = 0.5
HID_SIZE = 512

NUM_LAYERS = 3 # Number of hidden layers.

MAX_COLOR = 255

####### CNN Model params ########
CHANNEL_1 = 13
CHANNEL_2 = 5
