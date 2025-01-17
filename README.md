# Transfer Learning with Image Recognition CNNs for Sentiment Analysis
Developers: Nathaniel Nguyen, Chih Chen

## Proposal notes

### What we are trying to accomplish:
Our group would like to pursue a transfer learning project that would take
pre-trained models developed for image classification and repurpose these
models to perform sentiment classification on text input. The objective is to
see if we could take large open source models trained on copious amounts of
unrelated data and retrain these models to solve more niche problems, in our
case sentiment analysis.

Typically, transfer learning has been used to solve similarly related problems
such as transferring an image classification model to be used to solve a
different image classification problem. Instead, we are going to try and
repurpose image classification models to solve more disparate problems such as
sentiment analysis.

We are hoping that by using these repurposed models, we might be able to build
other models that would reduce training time, the number of required training
samples, or to reduce implementation complexity compared to training a model
traditionally from scratch. A successfully transferred model would would
indicate a viable mapping between two different modes of classification and
would open new opportunities to explore the effectiveness of other types of
model adaptations.

### How are you planning to do it?: 
We plan on using a pre-trained model: one from Google or AWS used for image
classification and train the final output layer of these models to solve the
sentiment analysis problem. 
The architectures we plan on using will most likely be CNN based model where
the convolution layers are pre-trained. However, we do not expect to fully know
what the exact architecture of the pre-built models may be implemented with.
Typically, these models will not be described in detail by the API, we plan on
using. 
The objective would be to remove or extend the final dense fully connected
layers to solve the sentiment analysis problem.
We also plan on building a simple CNN or RNN model to solve the problem using
the traditional "train a model from scratch" approach as our control
model. 
Some of the datasets we are thinking of using are as follows but not limited
to:
* Yelp (restaurant review) sentiment analysis 
* IMDB movie reviews
* Scrape data from other social media sites like Twitter, Amazon, or Facebook 
(maybe).

### How are you planning to evaluate whether you have succeeded?:
The test accuracy from the two approaches (transfer learning and "build
from scratch") will provide us with the metrics necessary to determine the
effectiveness of our implementations.

Furthermore, the training time and number of required samples will give us
further insight on the effectiveness of using transfer learning between
unrelated classification problems. 

## How to use:
Currently, this repo do not contain any trained checkpoints however, training a
simple model is trivial. 

To train a model head to ```LIGN167Project/src/models```. In here you should
see the file ```train.py``` This file is the main driver for training a model.
By default you can train a model by typing ```python train.py``` into the
terminal. If you would like, you can adjust the model by editing the
```Settings.py``` file located in ```LIGN167Project/src/helpers/```.

All logs from training will be outputed to ```LIGN167Project/src/data/logs```

## Description
For a detaileded post analysis, please refer to the accompaniment [paper](https://github.com/c-h-b-chen/Text_to_Image_Sentiment_Analysis/blob/master/Transfer%20Learning%20NLP.pdf)

## Acknowledgements
This project was conducted for the Deep Learning & Natural Language Processing course at the University of California, San Diego, in the fall of 2018 with Leon Bergen.
