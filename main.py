import tensorflow as tf
import numpy as np
from resnet import *
from tensorflow.examples.tutorials.mnist import input_data
from scipy.ndimage.interpolation import zoom

def inference(x, is_training):
    x = tf.reshape(x, [-1, 64, 64, 1])
    #model used is ResNet-34, modified to fit the tiny imagenet dataset
    with tf.variable_scope("conv1"):
        with tf.variable_scope("h1_conv_bn"):
            x = conv_wrapper(x, shape = [5,5,1,16], strides = [1, 1, 1, 1], padding = "VALID")
            x = bn_wrapper(x, is_training)
            x = tf.nn.relu(x)
        with tf.variable_scope("h2_conv_bn"):
            x = conv_wrapper(x, shape = [5,5,16,16], strides = [1, 1, 1, 1], padding = "VALID")
            x = bn_wrapper(x, is_training)
            x = tf.nn.relu(x)

    with tf.variable_scope("conv2_x"):
        # 3 residual blocks, 64
        blocks = 3
        channels = 16
        for n in range(blocks):
            with tf.variable_scope("residual_block_%d" % n):
                x = residual_block(x, channels, is_training)

    with tf.variable_scope("conv3_x"):
        # 4 residual blocks, 128
        blocks = 4
        channels = 24
        for n in range(blocks):
            with tf.variable_scope("residual_block_%d" % n):
                if n == 0:
                    x = residual_block_reduce_size(x, channels, is_training)
                else:
                    x = residual_block(x, channels, is_training)

    with tf.variable_scope("conv4_x"):
        # 6 residual blocks, 256
        blocks = 6
        channels = 32
        for n in range(blocks):
            with tf.variable_scope("residual_block_%d" % n):
                if n == 0:
                    x = residual_block_reduce_size(x, channels, is_training)
                else:
                    x = residual_block(x, channels, is_training)

    with tf.variable_scope("conv5_x"):
        # 3 residual blocks, 512
        blocks = 4
        channels = 48
        for n in range(blocks):
            with tf.variable_scope("residual_block_%d" % n):
                if n == 0:
                    x = residual_block_reduce_size(x, channels, is_training)
                else:
                    x = residual_block(x, channels, is_training)

    with tf.variable_scope("output"):
        #avgpool + something to get 10 classes
        with tf.variable_scope("avg_conv"):
            x = tf.nn.avg_pool(x, ksize = [1,6,6,1], strides = [1,6,6,1], padding = "VALID")
            x = conv_wrapper(x, shape = [1,1,48,10], strides = [1, 1, 1, 1], padding = "VALID")

    return tf.reshape(x, [-1, 10])

def loss(logits, labels):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = logits)
    return tf.reduce_mean(cross_entropy)

mnist = input_data.read_data_sets("MNIST_data", one_hot = True)
is_training = tf.placeholder(tf.bool)
x = tf.placeholder(tf.float32, shape = [None, 64*64*1])
y = tf.placeholder(tf.float32, shape = [None, 10])

logits = inference(x, is_training)
loss = loss(logits, y)
train_op = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss)
correct = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1600):
        batch = mnist.train.next_batch(100)
        image = batch[0]
        if i % 10 == 0:
            train_accuracy = accuracy.eval(feed_dict = {x: zoom(image, zoom = (1, (64/28)**2), order = 1),
                                                        y: batch[1],
                                                        is_training : True})
            print("step %d, training accuracy %f" % (i, train_accuracy))
        sess.run(train_op, feed_dict = {x : zoom(image, zoom = (1, (64/28)**2), order = 1),
                                        y : batch[1],
                                        is_training : True})

    batch_size = 1
    acc = []
    for i in range(int(10000/batch_size)):
        image = mnist.test.images[i*batch_size:(i+1)*batch_size]
        acc.append(accuracy.eval(feed_dict={x: zoom(image, zoom = (1, (64/28)**2), order = 1),
                                            y: mnist.test.labels[i*batch_size:(i+1)*batch_size],
                                            is_training : False}))
    acc = np.mean(acc)
    print("test accuracy", acc)
    print("done")
