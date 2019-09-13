
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

mnist=input_data.read_data_sets('MNIST', one_hot=True)


## Initialization
X = tf.compat.v1.placeholder(tf.float32, [None, 784])

# layer dimensions
L = 200
M = 100
N = 60
O = 30


# Probability of keeping a node during dropout = 1.0 at test time (no dropout) and 0.75 at training time
pkeep = tf.compat.v1.placeholder(tf.float32)

# Weights initialised with small random values between -0.2 and +0.2
# When using RELUs, make sure biases are initialised with small *positive* values for example 0.1 = tf.ones([K])/10
W1 = tf.Variable(tf.truncated_normal([784, L], stddev=0.1))  # 784 = 28 * 28
B1 = tf.Variable(tf.ones([L])/10)
W2 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
B2 = tf.Variable(tf.ones([M])/10)
W3 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
B3 = tf.Variable(tf.ones([N])/10)
W4 = tf.Variable(tf.truncated_normal([N, O], stddev=0.1))
B4 = tf.Variable(tf.ones([O])/10)
W5 = tf.Variable(tf.truncated_normal([O, 10], stddev=0.1))
B5 = tf.Variable(tf.zeros([10]))

# The model, with dropout at each layer
Y1 = tf.nn.relu(tf.matmul(X, W1) + B1)
Y1d = tf.nn.dropout(Y1, pkeep)

Y2 = tf.nn.relu(tf.matmul(Y1d, W2) + B2)
Y2d = tf.nn.dropout(Y2, pkeep)

Y3 = tf.nn.relu(tf.matmul(Y2d, W3) + B3)
Y3d = tf.nn.dropout(Y3, pkeep)

Y4 = tf.nn.relu(tf.matmul(Y3d, W4) + B4)
Y4d = tf.nn.dropout(Y4, pkeep)

Ylogits = tf.matmul(Y4d, W5) + B5
Y = tf.nn.softmax(Ylogits)


init = tf.compat.v1.global_variables_initializer()




# correct labels, one-hot encoding
Y_ = tf.compat.v1.placeholder(tf.float32, [None, 10])

## Loss
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

# evaluation
is_correct = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))


# Gradient Descent
optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.003)
train_step = optimizer.minimize(cross_entropy)



## Tensorflow Session
sess = tf.Session()
# sess.run to execute a node in the computation graph
sess.run(init)

test_data = {X:mnist.test.images, Y_:mnist.test.labels, pkeep: 1.0}


## Training loop
for i in range(1301):
    ## load batch of images and labels
    batch_X, batch_Y = mnist.train.next_batch(100)
    # dropout defined by percentage keep (pkeep)
    train_data={X: batch_X, Y_: batch_Y, pkeep: 0.75}


    # train
    sess.run(train_step, feed_dict=train_data)


    # eval
    if i % 50 == 0:
        a,c = sess.run([accuracy, cross_entropy], feed_dict=train_data)
        print(i, " train accuracy:\t", a, "cross entropy: ", c)
        a,c = sess.run([accuracy,cross_entropy],feed_dict = test_data)
        print(i, "test accuracy:\t", sess.run(accuracy, feed_dict=test_data))
        
#print (sess.run(W[350:360]))









