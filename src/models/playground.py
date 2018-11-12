import tensorflow as tf
import numpy as np

with tf.device('/device:GPU:0'):
    t1 = tf.constant(1.1, shape=[3, 3], name='Mytensor') 
    t2 = tf.constant(2.2, shape=[3,3], name='t2')
    t1nt2 = tf.add(t1, t2)


# Use this to track training and of models
# writer = tf.summary.FileWriter('.')
# writer.add_graph(tf.get_default_graph())
# writer.close()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(t1)
    print('t1nt2', t1nt2)
    print(sess.run({'test':t1, 'test2':t1nt2}))


#### Working with data

my_data = [
        [0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
        ]
slices = tf.data.Dataset.from_tensor_slices(my_data)
next_item = slices.make_one_shot_iterator().get_next()

with tf.Session() as sess:
    while True:
        try:
            print(sess.run(next_item))
        except tf.errors.OutOfRangeError:
            break

# Layers -- The building blocks for our models

x = tf.placeholder(tf.float32, shape=[None, 3])
linear_model = tf.layers.Dense(units=1)
y = linear_model(x)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(y, feed_dict={x: [[3,3,3]]}))


# Can use the feed_dict to replace a tensor.
print("another test")
xx = tf.constant(12)
yy = tf.square(xx)

with tf.Session() as sess:
    print(sess.run(yy, feed_dict={xx: 10}))

