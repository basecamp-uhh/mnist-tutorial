import math
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist=input_data.read_data_sets('MNIST', one_hot=True)


## Initialization, Input reshaped to 2-d image
X = tf.compat.v1.placeholder(tf.float32, [None, 784])
XX = tf.reshape(X, [-1, 28, 28, 1])

# three convolutional layers with their channel counts, and a
# fully connected layer (the last layer has 10 softmax neurons)
K = 4  # first convolutional layer output depth
L = 8  # second convolutional layer output depth
M = 12  # third convolutional layer
N = 200  # fully connected layer

W1 = tf.Variable(tf.truncated_normal([5, 5, 1, K], stddev=0.1))  # 5x5 patch, 1 input channel, K output channels
B1 = tf.Variable(tf.ones([K])/10)
W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))
B2 = tf.Variable(tf.ones([L])/10)
W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))
B3 = tf.Variable(tf.ones([M])/10)
W4 = tf.Variable(tf.truncated_normal([7 * 7 * M, N], stddev=0.1))
B4 = tf.Variable(tf.ones([N])/10)

W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
B5 = tf.Variable(tf.ones([10])/10)

# The model
stride = 1  # output is 28x28
Y1 = tf.nn.relu(tf.nn.conv2d(XX, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
stride = 2  # output is 14x14
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
stride = 2  # output is 7x7
Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)

# reshape the output from the third convolution for the fully connected layer
YY = tf.reshape(Y3, shape=[-1, 7 * 7 * M])

Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
Ylogits = tf.matmul(Y4, W5) + B5
Y = tf.nn.softmax(Ylogits)


# step for variable learning rate
step = tf.placeholder(tf.int32)


# correct labels, one-hot encoding
Y_ = tf.compat.v1.placeholder(tf.float32, [None, 10])

## Loss
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

# evaluation
is_correct = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# learning rate: exponential decay
lr = 0.0001 +  tf.train.exponential_decay(0.003, step, 2000, 1/math.e)

# Gradient Descent
optimizer = tf.compat.v1.train.AdamOptimizer(lr)
train_step = optimizer.minimize(cross_entropy)


init = tf.compat.v1.global_variables_initializer()



# Tensorflow Session
sess = tf.Session()
# sess.run to execute a node in the computation graph
sess.run(init)

test_data = {X: mnist.test.images, Y_:mnist.test.labels}


## Training loop
for i in range(10001):
    ## load batch of images and labels
    batch_X, batch_Y = mnist.train.next_batch(100)
    train_data={X: batch_X, Y_: batch_Y, step: i}


    # train
    sess.run(train_step, feed_dict=train_data)


    # eval
    if i % 50 == 0:
        a,c = sess.run([accuracy, cross_entropy], feed_dict=train_data)
        print(i, " train accuracy:\t", a, "cross entropy: ", c)
        a,c = sess.run([accuracy,cross_entropy],feed_dict = test_data)
        print(i, "test accuracy:\t", sess.run(accuracy, feed_dict=test_data))
        
#print (sess.run(W[350:360]))









