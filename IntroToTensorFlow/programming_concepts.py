# https://colab.research.google.com/notebooks/mlcc/tensorflow_programming_concepts.ipynb
import tensorflow as tf

# Tensors are arrays of arbitrary dimensionality
# * A scalar is a 0-d array (a 0th-order tensor)
# * A vector is a 1-d array (a 1st-order tensor). For example, [2, 3, 5, 7, 11] or [5]
# * A matrix is a 2-d array (a 2nd-order tensor). For example, [[3.1, 8.2, 5.9][4.3, -2.7, 6.5]]

# A TensorFlow graph is a graph data structure.  A graph's nodes are operations.
# Tensors flow through the graph, manipulated at each node by an operation.
# TensorFlow implements a lazy execution model.
# Tensors can be stored in the graph as constants or variables.

# Create a graph
g = tf.Graph()

# Establish the graph as the "default" graph.
with g.as_default():
  # Assemble a graph consisting of the following three operations:
  #   * Two tf.constant operations to create the operands.
  #   * One tf.add operation to add the two operands.
  x = tf.constant(8, name="x_const")
  y = tf.constant(5, name="y_const")
  my_sum = tf.add(x, y, name="x_y_sum")

  z = tf.constant(4, name="z_const")
  new_sum = tf.add(my_sum, z, name="x_y_z_sum")

  # Now create a session.
  # The session will run the default graph.
  with tf.Session() as sess:
    print(new_sum.eval())