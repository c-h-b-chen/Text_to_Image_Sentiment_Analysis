# This is the main driver for our project.
#import numpy as np
#import os 
#import pandas as pd
#import sys
import tensorflow as tf

#sys.path.insert(0, '../helpers/') 

#import LoadEmbeddings
#import MyInception
#import MyCNN
#import Settings

USE_TRANSFER = False

SAVE_MODEL = "data/checkpoint/Incep/Incep1.ckpt" if USE_TRANSFER else "data/checkpoint/CNN/CNN.ckpt"

NUM_WORDS = 75
EMB_DIM = 75

def run_demo():
    if (not tf.train.checkpoint_exists(SAVE_MODEL)):
        print("Sorry the model you want does not exist, ending program.")
        exit()

    device = "/device:GPU:0"

#    with tf.Session() as sess:
#        saver = tf.train.import_meta_graph(SAVE_MODEL+'.meta')
#        saver.restore(sess, tf.train.latest_checkpoint("data/checkpoint/CNN"))
##        builder = tf.saved_model.builder.SavedModelBuilder(SAVE_MODEL)
#
#        while True: 
#            review = input("Input some sentiment:\n>>")
#            if review == "":
#                break
#            print(review)
    with tf.device(device):
        # These are direct inputs into the training model.
        x = tf.placeholder(tf.float32, [None, NUM_WORDS, EMB_DIM, 3],
            name="all_sam")
        y = tf.placeholder(tf.int32, [None, ])

        # Build the model.
        scores = model_init_fn(x)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                logits=scores)

        loss = tf.reduce_mean(loss)

        optimizer = optimizer_init_fn()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss)

    best_val_acc = 0.0
    saver = tf.train.Saver()
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        saver.restore(sess, SAVE_MODEL)


if __name__ == "__main__":
    run_demo()


