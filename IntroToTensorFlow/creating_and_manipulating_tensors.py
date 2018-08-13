#https://colab.research.google.com/notebooks/mlcc/creating_and_manipulating_tensors.ipynb
import tensorflow as tf
try:
  tf.contrib.eager.enable_eager_execution()
  print("TF imported with eager execution!")
except ValueError:
  print("TF already imported with eager execution!")

# You can perform many typical mathematical operations on tensors (TF API).
# The code below creates the following vectors (1-D tensors), all having exactly six elements:

# * A primes vector containing prime numbers.
primes = tf.constant([2, 3, 5, 7, 11, 13], dtype=tf.int32)
print("primes:", primes)

# * A ones vector containing all 1 values.
ones = tf.ones([6], dtype=tf.int32)
print("ones:", ones)

# * A vector created by performing element-wise addition over the first two vectors.
just_beyond_primes = tf.add(primes, ones)
print("just_beyond_primes:", just_beyond_primes)

# * A vector created by doubling the elements in the primes vector.
twos = tf.constant([2, 2, 2, 2, 2, 2], dtype=tf.int32)
primes_doubled = primes * twos
print("primes_doubled:", primes_doubled)

# Printing a tensor returns not only its value, but also its shape (discussed in the next section) 
# and the type of value stored in the tensor. 
# Calling the numpy method of a tensor returns the value of the tensor as a numpy array:
some_matrix = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.int32)
print(some_matrix)
print("\nvalue of some_matrix is:\n", some_matrix.numpy())

# Shapes are used to characterize the size and number of dimensions of a tensor. 
# The shape of a tensor is expressed as list, with the ith element representing the size along dimension i. 
# The length of the list then indicates the rank of the tensor (i.e., the number of dimensions).

# A scalar (0-D tensor).
scalar = tf.zeros([])

# A vector with 3 elements.
vector = tf.zeros([3])

# A matrix with 2 rows and 3 columns.
matrix = tf.zeros([2, 3])

print('scalar has shape', scalar.get_shape(), 'and value:\n', scalar.numpy())
print('vector has shape', vector.get_shape(), 'and value:\n', vector.numpy())
print('matrix has shape', matrix.get_shape(), 'and value:\n', matrix.numpy())

# Broadcasting
# In mathematics, you can only perform element-wise operations (e.g. add and equals) on tensors of the same shape. 
# In TensorFlow, however, you may perform operations on tensors that would traditionally have been incompatible. 
# TensorFlow supports broadcasting (a concept borrowed from numpy), where the smaller array in an element-wise operation 
# is enlarged to have the same shape as the larger array.

# The following code performs the same tensor arithmetic as before, but instead uses 
# scalar values (instead of vectors containing all 1s or all 2s) and broadcasting.
primes = tf.constant([2, 3, 5, 7, 11, 13], dtype=tf.int32)
print("primes:", primes)

one = tf.constant(1, dtype=tf.int32)
print("one:", one)

just_beyond_primes = tf.add(primes, one)
print("just_beyond_primes:", just_beyond_primes)

two = tf.constant(2, dtype=tf.int32)
primes_doubled = primes * two
print("primes_doubled:", primes_doubled)

# Exercise #1: Arithmetic over vectors.
primes_squared = tf.pow(primes, 2)
print("primes_squared:", primes_squared)

just_under_primes_squared = tf.subtract(primes_squared, one)
print("just_under_primes_squared:", just_under_primes_squared)

# Matrix multiplication
# In linear algebra, when multiplying two matrices, the number of columns of the first matrix must equal the number of rows in the second matrix.

# A 3x4 matrix (2-d tensor).
x = tf.constant([[5, 2, 4, 3], [5, 1, 6, -2], [-1, 3, -1, -2]],
                dtype=tf.int32)

# A 4x2 matrix (2-d tensor).
y = tf.constant([[2, 2], [3, 5], [4, 5], [1, 6]], dtype=tf.int32)

# Multiply `x` by `y`; result is 3x2 matrix.
matrix_multiply_result = tf.matmul(x, y)

print(matrix_multiply_result)

# Tensor Reshaping
# With tensor addition and matrix multiplication each imposing constraints on operands, TensorFlow programmers must frequently reshape tensors.
# You can use the tf.reshape method to reshape a tensor. 
# For example, you can reshape a 8x2 tensor into a 2x8 tensor or a 4x4 tensor:

# Create an 8x2 matrix (2-D tensor).
matrix = tf.constant(
    [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]],
    dtype=tf.int32)

reshaped_2x8_matrix = tf.reshape(matrix, [2, 8])
reshaped_4x4_matrix = tf.reshape(matrix, [4, 4])

print("Original matrix (8x2):")
print(matrix.numpy())
print("Reshaped matrix (2x8):")
print(reshaped_2x8_matrix.numpy())
print("Reshaped matrix (4x4):")
print(reshaped_4x4_matrix.numpy())

# Exercise #2: Reshape two tensors in order to multiply them.

a = tf.constant([5, 3, 2, 7, 1, 4])
b = tf.constant([4, 6, 3])

reshaped_3x2_a = tf.reshape(a, [3, 2])
reshaped_1x3_b = tf.reshape(b, [1, 3])
c = tf.matmul(reshaped_1x3_b, reshaped_3x2_a)
print("c (1x2):")
print(c.numpy())

# Variables, Initialization and Assignment
# So far, all the operations we performed were on static values (tf.constant); calling numpy() always returned the same result. 
# TensorFlow allows you to define Variable objects, whose values can be changed.
# When creating a variable, you can set an initial value explicitly, or you can use an initializer (like a distribution):

