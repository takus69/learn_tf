from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import sys
import numpy as np
import parsebmp as pb

# Define method
def weight_variable(shape, name):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name=name)

def bias_variable(shape, name):
  initial= tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name=name)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def whatisit(file, sess):
  print("File name is %s" % file)
  data = pb.parse_bmp(file)

  # Show bmp data
  for i in range(len(data)):
    sys.stdout.write(str(int(data[i])))
    if (i+1) % 28 == 0:
      print("")

  # Predicting
  d = np.array([data])
  x_image = tf.reshape(d, [-1, 28, 28, 1])
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  h_pool1 = max_pool_2x2(h_conv1)
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
  h_pool2 = max_pool_2x2(h_conv2)
  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
  h_fc1_drop = tf.nn.dropout(h_fc1, 1.0)
  y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  result = sess.run(y_conv)

  # Show result
  print(result)
  print(np.argmax(result, 1))

if __name__ == "__main__":
  # Restore parameters
  W = tf.Variable(tf.zeros([28*28, 10]), name="W")
  b = tf.Variable(tf.zeros([10]), name="b")
  W_conv1 = weight_variable([5, 5, 1, 32], name="W_conv1")
  b_conv1 = bias_variable([32], name="b_conv1")
  W_conv2 = weight_variable([5, 5, 32, 64], name="W_conv2")
  b_conv2 = bias_variable([64], name="b_conv2")
  W_fc1 = weight_variable([7 * 7 * 64, 1024], name="W_fc1")
  b_fc1 = bias_variable([1024], name="b_fc1")
  W_fc2 = weight_variable([1024, 10], name="W_fc2")
  b_fc2 = bias_variable([10], name="b_fc2")

  sess = tf.InteractiveSession()
  saver = tf.train.Saver()
  saver.restore(sess, 'param/neural.param')

  # My data
  whatisit("My_data/0.bmp", sess)
  whatisit("My_data/1.bmp", sess)
  whatisit("My_data/2.bmp", sess)
  whatisit("My_data/3.bmp", sess)
  whatisit("My_data/4.bmp", sess)
  whatisit("My_data/5.bmp", sess)
  whatisit("My_data/6.bmp", sess)
  whatisit("My_data/7.bmp", sess)
  whatisit("My_data/8.bmp", sess)
  whatisit("My_data/9.bmp", sess)
