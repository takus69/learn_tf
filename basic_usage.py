import tensorflow as tf

# Graphs
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.], [2.]])
product = tf.matmul(matrix1, matrix2)

# Sessions
sess = tf.Session()
result = sess.run(product)
print(result)
# Output: [[12.]]
sess.close()


# Graphs
state = tf.Variable(0)
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

init_op = tf.initialize_all_variables()

# Sessions
with tf.Session() as sess:
  sess.run(init_op)
  print(sess.run(state))
  for _ in range(3):
    sess.run(state)
    print(sess.run(update))
# Output: 0 1 2 3


# Graphs
input1 = tf.constant([3.0])
input2 = tf.constant([2.0])
input3 = tf.constant([5.0])
intermed = tf.add(input2, input3)
mul = tf.mul(input1, intermed)

# Sessions
with tf.Session() as sess:
  result = sess.run([mul, intermed])
  print(result)
# Output: [array([ 21.], dtype=float32), array([ 7.], dtype=float32)]


# Graphs
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.mul(input1, input2)

# Sessions
with tf.Session() as sess:
  print(sess.run([output], feed_dict={input1:[7.], input2:[2.]}))
# Output: [array([ 14.], dtype=float32)]
