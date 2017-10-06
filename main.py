import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot = True)

def op(x, y, is_training):
    return tf.cond(is_training,
                   lambda: op_train_time(x,y),
                   lambda: op_test_time(x,y))

def op_train_time(x, y):
    return tf.mul(x, y)

def op_test_time(x, y):
    return tf.mul(x, tf.mul(tf.constant(100.0), y))

is_training = tf.placeholder(tf.bool)
num1 = tf.placeholder(tf.float32)
num2 = tf.placeholder(tf.float32)

op_test = op(num1, num2, is_training)

with tf.Session() as sess:
    output = sess.run(op_test, feed_dict = {num1 : 1.0, num2: 2.0, is_training : False})
    print(output)
