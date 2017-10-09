import tensorflow as tf
import numpy as np
from resnet import *
from tensorflow.examples.tutorials.mnist import input_data

def inference(x, is_training):
    reshaped_x = tf.reshape(x, [-1, 28, 28, 1])

    conv_input = conv_wrapper(reshaped_x, [1, 1, 1, 16], strides = [1,1,1,1], padding = "SAME")

    res1 = residual_block(conv_input, 16, is_training)
    res2 = residual_block(res1, 16, is_training)

    res3 = residual_block_reduce_size(res2, 32, is_training)
    res4 = residual_block(res3, 32, is_training)

    res5 = residual_block_reduce_size(res4, 48, is_training)
    res6 = residual_block(res5, 48, is_training)

    conv_10d = conv_wrapper(res6, [1, 1, 48, 10], strides = [1, 1, 1, 1], padding = "VALID")
    avg_pool = tf.nn.avg_pool(conv_10d, ksize = [1,6,6,1], strides = [1,6,6,1], padding = "VALID")

    logits = avg_pool
    return tf.reshape(logits, [-1, 10])

def loss(logits, labels):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = logits)
    return tf.reduce_mean(cross_entropy)

mnist = input_data.read_data_sets("MNIST_data", one_hot = True)
batch = mnist.train.next_batch(2)
print(batch[1].shape)
assert False == True

is_training = tf.placeholder(tf.bool)
x = tf.placeholder(tf.float32, shape = [None, 28*28*1])
y = tf.placeholder(tf.float32, shape = [None, 10])

logits = inference(x, is_training)
loss = loss(logits, y)
train_op = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(loss)
correct = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1600):
        batch = mnist.train.next_batch(100)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict = {x: batch[0], y: batch[1], is_training : True})
            print("step %d, training accuracy %f" % (i, train_accuracy))
        sess.run(train_op, feed_dict = {x : batch[0], y : batch[1], is_training : True})

    batch_size = 100
    acc = []
    for i in range(int(10000/batch_size)):
        acc.append(accuracy.eval(feed_dict={x: mnist.test.images[i*batch_size:(i+1)*batch_size], y: mnist.test.labels[i*batch_size:(i+1)*batch_size], is_training : False}))
    acc = np.mean(acc)
    print("test accuracy", acc)
    print("done")

# def test(x, is_training):
#     x_reshaped = tf.reshape(x, [-1,28,28,1])
#     d = x_reshaped.get_shape().as_list()[3]
#     num_channels = 31
#     conv = conv_wrapper(x_reshaped, shape = [3,3,d,num_channels], strides = [1, 1, 1, 1], padding = "SAME")
#     return conv
# op = test(x, is_training)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     batch = mnist.train.next_batch(50)
#     output = sess.run(op, feed_dict = {x: batch[0], is_training: True})
#     print(output.shape)
