import tensorflow as tf
import numpy as np
from resnet import *
from tensorflow.examples.tutorials.mnist import input_data

def inference(x):

    #TODO BUILD THE RESNET HERE




    #ENDTODO

    return logits

def loss(logits, labels):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = logits)
    return tf.reduce_mean(cross_entropy)

mnist = input_data.read_data_sets("MNIST_data", one_hot = True)

is_training = tf.placeholder(tf.bool)
num1 = tf.placeholder(tf.float32)
num2 = tf.placeholder(tf.float32)

op_test = op(num1, num2, is_training)

with tf.Session() as sess:
    output = sess.run(op_test, feed_dict = {num1 : 1.0, num2: 2.0, is_training : False})
    print(output)
