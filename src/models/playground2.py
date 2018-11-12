import tensorflow as tf
import numpy as np
# Build a simple linear regression model


# Define some data
x_data = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)

# Build the model
linear_model = tf.layers.Dense(units=1)
y_pred = linear_model(x_data)

# Run the model
sess = tf.Session()
init = tf.global_variables_initializer()

sess.run(init)
print(sess.run(y_pred))

# Determine the loss of the model
loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
print(sess.run(loss))

# Train the model
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
for _ in range(1000):
    _, loss_value = sess.run((train, loss))
    # print(loss_value)

print(sess.run(y_true))
print(sess.run(y_pred))
