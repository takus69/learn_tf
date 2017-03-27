from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import sys
import numpy as np
import parsebmp as pb

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
  result = sess.run(y, feed_dict={x: d})

  # Show result
  print(result)
  print(np.argmax(result, 1))


if __name__ == "__main__":
  # Restore parameters
  x = tf.placeholder(tf.float32, shape=[None, 28*28])
  y_ = tf.placeholder(tf.float32, shape=[None, 10])
  W = tf.Variable(tf.zeros([28*28, 10]), name="W")
  b = tf.Variable(tf.zeros([10]), name="b")
  y = tf.matmul(x, W) + b
  
  sess = tf.InteractiveSession()
  saver = tf.train.Saver()
  saver.restore(sess, 'param/softmax.param')

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
