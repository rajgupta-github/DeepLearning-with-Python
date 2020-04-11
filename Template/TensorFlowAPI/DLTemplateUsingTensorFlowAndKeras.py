# Packages Installation related to Tensorflow
# !pip install tensorflow-gpu==1.13.1
# !pip install tensorflow-gpu==2.0.0-alpha0

#Import Packages command
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np

# Declaring Constant, Variable , String in Tensorflow
tf2_constant = tf.constant([[1, 2], [3, 4]])
tf2_variable = tf.Variable([[1., 2., 3.], [4., 5., 6.]])
tf2_string = tf.constant("TensorFlow")


