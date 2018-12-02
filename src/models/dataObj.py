import numpy as np
import tensorflow as tf

#a = np.zeros((3, [x for x in range(2)]))

a = np.zeros((25000, 3, 299))

print(a.shape)

print(type(a), a.shape)
print(type(a[0]), a[0].shape)
print(type(a[0][0]), a[0][0].shape)
print(type(a[0][0][0]))

x_training = a
y_training = np.zeros((25000,))

# Weights must be on cpu because embedding doesn't support GPU
with tf.device('cpu:0'):
    # Features used to train. Contains 3 channels. Must convert all 3
    inputs = tf.placeholder(tf.int32, [None], name='word_ids')

    x_train = tf.placeholder(tf.int32, shape=[None, 3, 299])
    y_train = tf.placeholder(tf.int32, shape=[None,])

    # Build a dataset
    train_dset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_it = train_dset.make_initializable_iterator()
    train_x_el, train_y_el = train_it.get_next()

    # Load the emebedding the cpu
#    W1 = tf.constant(wv_m1.syn0, name="W1")
#    W2 = tf.constant(wv_m1.syn0, name="W2")
#    W3 = tf.constant(wv_m1.syn0, name="W3")

    # Tensor that will perform the conversion to true embedding vector. 
    # inputs should be of shape (299,)
#        embedded1 = tf.nn.embedding_lookup(W1, inputs)
#        embedded2 = tf.nn.embedding_lookup(W2, inputs)
#        embedded3 = tf.nn.embedding_lookup(W3, inputs)

#    embedded1 = tf.nn.embedding_lookup(W1, train_x_el)
#    embedded2 = tf.nn.embedding_lookup(W2, train_x_el)
#    embedded3 = tf.nn.embedding_lookup(W3, train_x_el)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(train_it.initializer, feed_dict={x_train: x_training, y_train: y_training})
