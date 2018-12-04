# File directory
LoadIMDB.py -- Load/Build pickles for the IMDB dataset.

LoadYelp.py -- Load the Yelp dataset.

VanillaRNN.py -- A simple non-pre-training RNN model.

# Datasets
Dataset | Num samples

Train Positive -- 12500
Train Negative -- 12500

Val Positive -- 2500
Val Negative -- 2500

Train Positive -- 10000
Train Negative -- 10000

* Took 5 samples from the test set to use as the validation set.

# W2V
Changed window size for each of the word2vec models. Hopefully it will promote
different amount of variance.


# Ideas/TODOS

* Pre-embed all data for faster train computation time? ATM very slow.
* Check if we need to scale the embeddings from word2vec
* Add support for larger batch size in training.
