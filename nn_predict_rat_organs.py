import pandas as pd
import numpy as np
from scipy import stats as scistats
#import matplotlib.pyplot as plt
#import matplotlib.mlab as mlab
from scipy.sparse import csr_matrix
from sklearn.cross_validation import train_test_split
#from CTM import CTM
#from CTMParallel import CTMParallel
import csv
import sys
import pickle

#if len(sys.argv) != 2:#
#	print "Usage: RunCTM_ratorgans.py numtopics"
#	sys.exit()
data = pd.read_csv('counts_rat.csv').set_index('GeneID')
genes_wanted = (data > 1).sum(axis=1) > 5
genes_wanted = data.var(axis=1).sort(inplace=False, ascending=False)[:1000].index
counts_newsetup = data.ix[genes_wanted, :].T
classes = np.array(map(lambda x: x.split("_")[0], counts_newsetup.index))
classes = pd.get_dummies(classes).values
vocab = counts_newsetup.columns
counts = counts_newsetup.values.astype(float)

trX, teX, trY, teY  = train_test_split(counts, classes, test_size=0.2, random_state=42)

import tensorflow as tf
sess = tf.InteractiveSession()

print trX.shape, trY.shape
print trY

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

# Model with 2 hidden layers
def model_2hidden(X, w_h, w_h2, w_o, b_h, b_h2, b_o, p_drop_input, p_drop_hidden): # this network is the same as the previous one except with an extra hidden layer + dropout
    #X = tf.nn.dropout(X, p_drop_input)
    h = tf.nn.relu(tf.matmul(X, w_h) + b_h)

    #h = tf.nn.dropout(h, p_drop_hidden)
    h2 = tf.nn.relu(tf.matmul(h, w_h2) + b_h2)

    #h2 = tf.nn.dropout(h2, p_drop_hidden)

    return tf.matmul(h2, w_o) + b_o

# Model with 1 hidden layer
def model_1hidden(X, w_h, w_o, b_h, b_o, p_drop_input, p_drop_hidden): # this network is the same as the previous one except with an extra hidden layer + dropout
    #X = tf.nn.dropout(X, p_drop_input)
    h = tf.nn.relu(tf.matmul(X, w_h) + b_h)
    #h = tf.nn.dropout(h, p_drop_hidden)

    return tf.matmul(h, w_o) + b_o

model = model_2hidden

# Placeholders for input image and output classes
X = tf.placeholder("float", shape=[None, 1000])
Y = tf.placeholder("float", shape=[None, 11])

w_h = init_weights([1000, 625])
b_h = init_weights([625])


w_h2 = init_weights([625, 625])
b_h2 = init_weights([625])

w_o = init_weights([625, 11])
b_o = init_weights([11])

p_keep_input = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")

#py_x = model_2hidden(X, w_h, w_h2, w_o, b_h, b_h2, b_o, p_keep_input, p_keep_hidden)
py_x = model_1hidden(X, w_h, w_o, b_h, b_o, p_keep_input, p_keep_hidden)


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
        sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                      p_keep_input: 0.8, p_keep_hidden: 0.5})
    print i, np.mean(np.argmax(teY, axis=1) ==
                     sess.run(predict_op, feed_dict={X: teX, Y: teY,
                                                     p_keep_input: 1.0,
                                                     p_keep_hidden: 1.0}))
