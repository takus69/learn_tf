from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import sys
import numpy as np
import parsebmp as pb

# Download gz files to MNIST_data directory
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
batch = mnist.train.next_batch(100)

# Initializing
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 28*28])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([28*28, 10]))
b = tf.Variable(tf.zeros([10]))

sess.run(tf.initialize_all_variables())

# Making model
y = tf.matmul(x, W) + b
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

# Training
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
for i in range(1000):
  #if i % 100 == 0:
  #  print "i = " + str(i)
  batch = mnist.train.next_batch(100)
#  for i in range(len(batch[0])):
#    for j in range(len(batch[0][i])):
#      if batch[0][i][j] > 0:
#        batch[0][i][j] = 1
#      else:
#        batch[0][i][j] = 0
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

# Evaluating
#correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


def whatisit(file):
  data = pb.parse_bmp(file)

  # Output data
  for i in range(len(data)):
#    if data[i] > 0:
#      sys.stdout.write("1")
#    else:
#      sys.stdout.write("0")
    sys.stdout.write(str(int(data[i])))
    if (i+1) % 28 == 0:
      print("")

  # Predicting
  d = np.array([data])
  result = sess.run(tf.nn.softmax(tf.matmul(d, W) + b))
  print(result)
  print(np.argmax(result))

if __name__ == "__main__":
  # Mnist data
  data = batch[0][1]
  for i in range(len(data)):
    if data[i] > 0:
#      sys.stdout.write("1")
      data[i] = 1
    else:
#      sys.stdout.write("0")
      data[i] = 0
    sys.stdout.write(str(int(data[i])))
    if (i+1) % 28 == 0:
      print("")
  d = np.array([data])
  result = sess.run(tf.nn.softmax(tf.matmul(d, W) + b))
  print(result)
  print("predict = " + str(np.argmax(result)))
  print("actual = " + str(np.argmax(batch[1][1])))
  
  # My data
  whatisit("0.bmp")
  whatisit("1.bmp")
  whatisit("2.bmp")
  whatisit("3.bmp")
  whatisit("4.bmp")
  whatisit("5.bmp")
  whatisit("6.bmp")
  whatisit("7.bmp")
  whatisit("8.bmp")
  whatisit("9.bmp")
