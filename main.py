import tensorflow as tf
import numpy as np
from resnet import *
from data_loader import *
from scipy.ndimage.interpolation import zoom

def inference(x, is_training):
    #model used is ResNet-18, modified to fit the tiny imagenet dataset
    with tf.variable_scope("conv1"):
        with tf.variable_scope("h1_conv_bn"):
            x = conv_wrapper(x, shape = [3,3,3,64], strides = [1, 2, 2, 1], padding = "VALID")
            x = bn_wrapper(x, is_training)
            x = tf.nn.relu(x)

    with tf.variable_scope("conv2_x"):
        # 2 residual blocks, 64
        channels = 64
        with tf.variable_scope("residual_block_1"):
            x = residual_block(x, channels, is_training)
        with tf.variable_scope("residual_block_2"):
            x = residual_block(x, channels, is_training)

    with tf.variable_scope("conv3_x"):
        # 2 residual blocks, 128
        channels = 128
        with tf.variable_scope("residual_block_1"):
            x = residual_block_reduce_size(x, channels, is_training)
        with tf.variable_scope("residual_block_2"):
            x = residual_block(x, channels, is_training)

    with tf.variable_scope("conv4_x"):
        # 2 residual blocks, 192
        channels = 192
        with tf.variable_scope("residual_block_1"):
            x = residual_block_reduce_size(x, channels, is_training)
        with tf.variable_scope("residual_block_2"):
            x = residual_block(x, channels, is_training)

    with tf.variable_scope("conv5_x"):
        # 2 residual blocks, 256
        channels = 256
        with tf.variable_scope("residual_block_1"):
            x = residual_block_reduce_size(x, channels, is_training)
        with tf.variable_scope("residual_block_2"):
            x = residual_block(x, channels, is_training)

    with tf.variable_scope("output"):
        #avgpool + something to get 10 classes
        with tf.variable_scope("avg_conv"):
            x = tf.nn.avg_pool(x, ksize = [1,3,3,1], strides = [1,3,3,1], padding = "VALID")
            x = conv_wrapper(x, shape = [1,1,256,200], strides = [1, 1, 1, 1], padding = "VALID")

    return tf.reshape(x, [-1, 200])

def loss(logits, labels):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = logits)
    return tf.reduce_mean(cross_entropy)

is_training = tf.placeholder(tf.bool)
x = tf.placeholder(tf.float32, shape = [None, 64, 64, 3])
y = tf.placeholder(tf.float32, shape = [None, 200])

data_loader = data_loader()
data_loader.load_data_arrays()
#data_loader.shuffle_data()
str_to_class_lookup = data_loader.str_to_class_lookup
class_to_str_lookup = data_loader.class_to_str_lookup

logits = inference(x, is_training)
loss = loss(logits, y)
train_op = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(loss)
correct = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

tf.summary.scalar("loss", loss)

summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("testpath", sess.graph)
    for i in range(100):
        batch = data_loader.next_batch(10)
        image = batch[0]
        if i % 10 == 0:
            train_accuracy = accuracy.eval(feed_dict = {x: batch[0],
                                                        y: batch[1],
                                                        is_training : True})
            print("step %d, training accuracy %f" % (i, train_accuracy))
        _, summary = sess.run([train_op, summary_op], feed_dict = {x : batch[0],
                                                                   y : batch[1],
                                                                   is_training : True})

        writer.add_summary(summary, i * 10)
    #test validation data
    # batch_size = 1
    # acc = []
    # for i in range(int(10000/batch_size)):
    #     image = mnist.test.images[i*batch_size:(i+1)*batch_size]
    #     acc.append(accuracy.eval(feed_dict={x: zoom(image, zoom = (1, (64/28)**2), order = 1),
    #                                         y: mnist.test.labels[i*batch_size:(i+1)*batch_size],
    #                                         is_training : False}))
    # acc = np.mean(acc)
