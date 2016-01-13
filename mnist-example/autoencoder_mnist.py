# Some of this code is based on the simple tensorflow neural network example found at https://github.com/nlintz/TensorFlow-Tutorials/blob/master/4_modern_net.py
import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

import tensorflow as tf
sess = tf.InteractiveSession()

print trX.shape, trY.shape
print trY

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w_h, w_h2, w_o, b_h, b_h2, b_o, p_drop_input, p_drop_hidden): # this network is the same as the previous one except with an extra hidden layer + dropout
    #X = tf.nn.dropout(X, p_drop_input)
    h = tf.nn.relu(tf.matmul(X, w_h) + b_h)

    #h = tf.nn.dropout(h, p_drop_hidden)
    h2 = tf.nn.relu(tf.matmul(h, w_h2) + b_h2)

    #h2 = tf.nn.dropout(h2, p_drop_hidden)

    return tf.matmul(h2, w_o) + b_o

# Placeholders for input image and output classes
X = tf.placeholder("float", shape=[None, 784])
Y = tf.placeholder("float", shape=[None, 784])

w_h = init_weights([784, 625])
b_h = init_weights([625])


w_h2 = init_weights([625, 625])
b_h2 = init_weights([625])

w_o = init_weights([625, 784])
b_o = init_weights([784])

p_keep_input = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")

py_x = model(X, w_h, w_h2, w_o, b_h, b_h2, b_o, p_keep_input, p_keep_hidden)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
train_op = tf.train.AdagradOptimizer(0.001, 0.9).minimize(cost)

#cost = -tf.reduce_sum(Y*tf.log(tf.nn.softmax(py_x)))
#train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)


predict_op = tf.argmax(py_x, 1)


sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)


for i in range(100):
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
        sess.run(train_op, feed_dict={X: trX[start:end], Y: trX[start:end],
                                      p_keep_input: 0.8, p_keep_hidden: 0.5})
    print i, np.mean(np.argmax(teX, axis=1) ==
                     sess.run(predict_op, feed_dict={X: teX, Y: teX,
                                                     p_keep_input: 1.0,
                                                     p_keep_hidden: 1.0}))














