import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
import tensorflow as tf
from sklearn.preprocessing import Binarizer


df_input = pd.read_csv("C:/Users/Administrator/Desktop/stockpredict/input.csv", encoding='gbk')
df_label = pd.read_csv("C:/Users/Administrator/Desktop/stockpredict/label.csv", encoding='gbk')
# 获取[1,0]格式的label
def get_label(label, position):
    df_label.ix[:, position] = label.ix[:, position].str.strip("%").astype(float)/100
    label1 = np.array(label.ix[:, position])[:, np.newaxis]
    label2 = np.array(label.ix[:, position])[:, np.newaxis]
    label1[label1 > 0] = 1
    label1[label1 < 0] = 0
    label2[label2 > 0] = 0
    label2[label2 < 0] = 1
    y_data = np.concatenate((label1, label2), axis=1)[1:]
    return y_data[:246], y_data[246:]

# 获取格式好的input
def get_input(input):
    x_data = scale(np.array(input.ix[:, 1:8], dtype=str).astype(float), axis=0)[:-1]
    return x_data[:246], x_data[246:]

# def batch_normalization(Wx_plus_b,out_size):
#     fc_mean, fc_var = tf.nn.moments(
#         Wx_plus_b,
#         axes=[0],  # the dimension you wanna normalize, here [0] for batch
#         # for image, you wanna do [0, 1, 2] for [batch, height, width] but not channel
#     )
#     scale = tf.Variable(tf.ones([out_size]))
#     shift = tf.Variable(tf.zeros([out_size]))
#     epsilon = 0.001
#
#     # apply moving average for mean and var when train on batch
#     ema = tf.train.ExponentialMovingAverage(decay=0.5)
#
#     def mean_var_with_update():
#         ema_apply_op = ema.apply([fc_mean, fc_var])
#         with tf.control_dependencies([ema_apply_op]):
#             return tf.identity(fc_mean), tf.identity(fc_var)
#
#     mean, var = mean_var_with_update()
#
#     Wx_plus_b = tf.nn.batch_normalization(Wx_plus_b, mean, var, shift, scale, epsilon)
#     # similar with this two steps:
#     # Wx_plus_b = (Wx_plus_b - fc_mean) / tf.sqrt(fc_var + 0.001)
#     # Wx_plus_b = Wx_plus_b * scale + shift
#
#     # activation
#     outputs = tf.nn.relu(Wx_plus_b)
#
#     return outputs

# set x,y placeholder


tf_x = tf.placeholder(tf.float32, [None, 6])
tf_y = tf.placeholder(tf.float32, [None, 2])

# neural network layers


def add_input_layer(x):
    # y = tf.layers.dense(x, 12)
    # return batch_normalization(y, 12)
    return tf.layers.dense(x, 12, activation=tf.nn.relu)


in_result = add_input_layer(tf_x)


def add_hidden_layer1(x):
    # y = tf.layers.dense(x, 24)
    # return batch_normalization(y, 24)
    return tf.layers.dense(x, 24, activation=tf.nn.relu)


h1_result = add_hidden_layer1(in_result)


def add_hidden_layer2(x):
    # y = tf.layers.dense(x, 12)
    # return batch_normalization(y, 12)
    return tf.layers.dense(x, 12, activation=tf.nn.relu)


h2_result = add_hidden_layer2(h1_result)


def add_out_layer(x):
    return tf.layers.dense(x, 2, activation=tf.nn.softmax)


prediction = add_out_layer(h2_result)

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf_y * tf.log(prediction),
                                              reduction_indices=[1]  #loss
                                              ))

train_step = tf.train.AdamOptimizer(0.3).minimize(cross_entropy)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

x_train, x_test = get_input(df_input)
y_train, y_test = get_label(df_label, 1)
sess.run(train_step, feed_dict={tf_x: x_train, tf_y: y_train})

y_pre = sess.run(prediction, feed_dict={tf_x: x_test})

correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(y_test, 1))

print(sess.run(correct_prediction))
