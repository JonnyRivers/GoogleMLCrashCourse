# https://colab.research.google.com/notebooks/mlcc/hello_world.ipynb
import tensorflow as tf
try:
  tf.contrib.eager.enable_eager_execution()
except ValueError:
  pass  # enable_eager_execution errors after its first call

tensor = tf.constant('Hello, world!')
tensor_value = tensor.numpy()
print(tensor_value)