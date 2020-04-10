import tensorflow as tf
import numpy as np
# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



print("printing tf version", tf.__version__)

print("==========TF Constants=============")

tensor_20 = tf.constant([[23, 4], [32, 51]])
print("printing tf constant object" , tensor_20)

# Getting the shape of a tensor
print("printing tensor constant shape" , tensor_20.shape)

# Getting the values straight from a TensorFlow constant with numpy, and without the need of a session
print("printing tensor constant with numpy", tensor_20.numpy())

# We are able to convert a numpy array back to a TensorFlow tensor as well
numpy_tensor = np.array([[23,  4], [32, 51]])
print("printing numpy tensor", numpy_tensor)

tensor_from_numpy = tf.constant(numpy_tensor)
print("printing tensor from numpy", tensor_from_numpy)

print("==========TF Variables=============")

tf2_variable = tf.Variable([[1., 2., 3.], [4., 5., 6.]])
print("printing variable tf object", tf2_variable)

print("Getting values from numpy", tf2_variable.numpy())

tf2_variable[0, 2].assign(100)
print("Getting values from numpy after assignment", tf2_variable.numpy())

x=tf.Variable(3)
y=tf.Variable(5)
z=x+y
print("printing addition of 2 variables", z.numpy())


print("==========Operations with tensors=============")

tensor = tf.constant([[1, 2], [3, 4]])
print("original tensor", tensor.numpy())
print("printing after addition", tensor+2)
print("printing after multiplication", tensor*5)

# Getting the squares of all numbers in a TensorFlow tensor object
print("squaring of tensor", np.square(tensor))

# Getting the square root of all numbers in a tensorflow tensor object
print("square root of tensor", np.sqrt(tensor))

print("dot product between 2 tensors",  np.dot(tensor, tensor_20))

print("==========TF Strings=============")
tf_string = tf.constant("TensorFlow")
print("printing tf string", tf_string)

print("printing string length", tf.strings.length(tf_string).numpy())

print("printing unicode encoding of a string", tf.strings.unicode_decode(tf_string, "UTF8").numpy())

tf_string_array = tf.constant(["TensorFlow", "Deep Learning", "AI"])
# Iterating through the TF string array
print("Printing Array of strings")
for string in tf_string_array:
  print(string.numpy())