import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

x  = tf.placeholder(tf.float32, shape=[None, 784])
yt = tf.placeholder(tf.float32, shape=[None, 10])
W  = tf.Variable(tf.zeros([784, 10]))
b  = tf.Variable(tf.zeros([10]))
y  = tf.matmul(x, W) + b
ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=yt, logits=y))
ts = tf.train.GradientDescentOptimizer(0.5).minimize(ce)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for step in range(3001):
    bx, byt = mnist.train.next_batch(100)
    _, loss = sess.run([ts, ce], feed_dict={x: bx, yt: byt})
    if step % 100 == 0:
        print("steps: {0:4d}, loss: {1}".format(step, loss))
    #ts.run(feed_dict={x: bx, yt: byt})
cp = tf.equal(tf.argmax(y, 1), tf.argmax(yt, 1))
ac = tf.reduce_mean(tf.cast(cp, tf.float32))

print(ac.eval(feed_dict={x: mnist.test.images, yt: mnist.test.labels}))
