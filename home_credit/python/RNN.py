import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def transform_label(DFrame1,DFrame2):
    for i in DFrame1.columns:
        if isinstance(DFrame1.loc[0,i],str):
            print(i)
            train[i] = train[i].fillna('nan')
            test[i] = test[i].fillna('nan')
            tmp = list(DFrame1[i])
            tmp.extend(DFrame2[i])
            label = LabelEncoder().fit(tmp)
            DFrame1[i] = label.transform(DFrame1[i])
            DFrame2[i] = label.transform(DFrame2[i])
            #tmp = list(DFrame1[i])
            #tmp.extend(DFrame2[i])
            
train = pd.read_csv('../data/application_train.csv')
test = pd.read_csv('../data/application_test.csv')
transform_label(train,test)

lr = 0.001
training_iters = 10000
batch_size = train.shape[0]
n_inputs = train.shape[1]-1
n_steps = 1
n_hidden_units = 128
n_classes = 2

x = tf.placeholder(tf.float32,[None,n_steps,n_inputs])
y = tf.placeholder(tf.float32,[None,n_classes])

weights = {'in':tf.Variable(tf.random_normal([n_inputs,n_hidden_units])),
           'out':tf.Variable(tf.random_normal([n_hidden_units,n_classes]))}
biases = {'in':tf.Variable(tf.constant(0.1,shape=[n_hidden_units,])),
          'out':tf.Variable(tf.constant(0.1,shape=[n_classes,]))}

def RNN(X,weights,biases):
    X = tf.reshape(X, [-1, n_inputs])
    X_in = tf.matmul(X,weights['in'])+biases['in']
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units,forget_bias=1.0,state_is_tuple=True)
    init_state = lstm_cell.zero_state(batch_size,dtype=tf.float32)
    outputs,states = tf.nn.dynamic_rnn(lstm_cell,X_in,initial_state=init_state,time_major=False)
    results = tf.matmul(states[1],weights['out'])+biases['out']
    return results

pred = RNN(x,weights,biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step*batch_size < training_iters:
        batch_xs,batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={x:batch_xs, y:batch_ys})
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={x:batch_xs, y:batch_ys}))
        step += 1


