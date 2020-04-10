import tensorflow as tf
print(tf.__version__)

tensor_20 = tf.constant([[23, 4], [32, 51]])
print("printing object" , tensor_20)

# Getting the shape of a tensor
print("printing tensor shape" , tensor_20.shape)

# Getting the values straight from a TensorFlow constant with numpy, and without the need of a session
tensor_20.numpy()