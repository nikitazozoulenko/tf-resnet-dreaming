import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot = True)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W):
    #x has shape (R, W, H, D)
    return tf.nn.conv2d(x,
                        W,
                        strides = [1, 1, 1, 1],
                        padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x,
                          ksize = [1, 2, 2, 1],
                          strides = [1,2,2,1],
                          padding = "SAME")

def relu(x):
    return tf.nn.relu(x)

EPSILON = 0.00001

def bn(x, beta, gamma, moving_mean, moving_variance, is_training):
    return tf.cond(is_training,
                   lambda: bn_train_time(x, beta, gamma, moving_mean, moving_variance),
                   lambda: bn_test_time(x, beta, gamma, moving_mean, moving_variance))

def bn_train_time(x, beta, gamma, moving_mean, moving_variance):
    mean, variance = tf.nn.moments(x, axes = [0,1,2])
    op_moving_mean = tf.assign(moving_mean,
                               moving_mean * 0.9 + mean * (0.1))
    op_moving_variance = tf.assign(moving_variance,
                                   moving_variance * 0.9 + variance * (0.1))
    with tf.control_dependencies([op_moving_mean, op_moving_variance]):
        return tf.nn.batch_normalization(x,
                                         mean,
                                         variance,
                                         offset = beta,
                                         scale = gamma,
                                         variance_epsilon = EPSILON)

def bn_test_time(x, beta, gamma, moving_mean, moving_variance):
    return tf.nn.batch_normalization(x,
                                     moving_mean,
                                     moving_variance,
                                     offset = beta,
                                     scale = gamma,
                                     variance_epsilon = EPSILON)


x = tf.placeholder(tf.float32, shape = [None, 28*28*1])
y = tf.placeholder(tf.float32, shape = [None, 10])
is_training = tf.placeholder(tf.bool)
x_image = tf.reshape(x, [-1, 28, 28, 1])

W_conv1 = weight_variable([5,5,1,32]) #weights have shape [W,H,D,num_channels]
b_conv1 = bias_variable([32]) #biases have shape [num_channels]
bn1_gamma = tf.Variable(tf.constant(1.0, shape = [32]))
bn1_beta = tf.Variable(tf.constant(0.0, shape = [32]))
bn1_moving_mean = tf.Variable(tf.constant(0.0, shape = [32]), trainable = False)
bn1_moving_variance = tf.Variable(tf.constant(1.0, shape = [32]), trainable = False)

h_conv1 = relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
bn1 = bn(h_pool1, bn1_beta, bn1_gamma, bn1_moving_mean, bn1_moving_variance, is_training)

W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
bn2_gamma = tf.Variable(tf.constant(1.0, shape = [64]))
bn2_beta = tf.Variable(tf.constant(0.0, shape = [64]))
bn2_moving_mean = tf.Variable(tf.constant(0.0, shape = [64]), trainable = False)
bn2_moving_variance = tf.Variable(tf.constant(1.0, shape = [64]), trainable = False)

h_conv2 = relu(conv2d(bn1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
bn2 = bn(h_pool2, bn2_beta, bn2_gamma, bn2_moving_mean, bn2_moving_variance, is_training)

W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(bn2, [-1, 7*7*64])
h_fc1 = relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_hat = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = y_hat))
train_step = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        batch = mnist.train.next_batch(100)
        if i % 10 == 0:
            train_accuracy = accuracy.eval(feed_dict = {x: batch[0], y: batch[1], is_training : True, keep_prob : 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        sess.run(train_step, feed_dict = {x : batch[0], y : batch[1], is_training : True, keep_prob : 0.5})
        #train_step.run(feed_dict = {x : batch[0], y : batch[1], keep_prob : 0.5})

    test_accuracy = accuracy.eval(feed_dict={x: mnist.test.images[:100], y: mnist.test.labels[:100], is_training : False, keep_prob: 1.0})
    print("test accuracy %g" % test_accuracy)
    print("done")
