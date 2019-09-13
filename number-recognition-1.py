
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

mnist=input_data.read_data_sets('MNIST', one_hot=True)


## Network initialization
X = tf.compat.v1.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

init = tf.compat.v1.global_variables_initializer()




## Model
# classification of input images
# matrix multiplication: X[100, 784] W[784, 10] b[10]
Y = tf.nn.softmax(tf.matmul(X, W) + b)

# correct labels, one-hot encoding
Y_ = tf.compat.v1.placeholder(tf.float32, [None, 10])

## Loss
cross_entropy = -tf.reduce_sum(Y_ * tf.math.log(Y))

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

test_data = {X:mnist.test.images, Y_:mnist.test.labels}


## Training loop
for i in range(1001):
    ## load batch of images and labels
    batch_X, batch_Y = mnist.train.next_batch(100)
    train_data={X: batch_X, Y_: batch_Y}


    # train
    sess.run(train_step, feed_dict=train_data)


    # eval
    if i % 50 == 0:
        a,c = sess.run([accuracy, cross_entropy], feed_dict=train_data)
        print(i, " train accuracy:\t", a, "cross entropy: ", c)
        a,c = sess.run([accuracy,cross_entropy],feed_dict = test_data)
        print(i, "test accuracy:\t", sess.run(accuracy, feed_dict={X: mnist.test.images, Y_: mnist.test.labels}))
        
#print (sess.run(W[350:360]))









