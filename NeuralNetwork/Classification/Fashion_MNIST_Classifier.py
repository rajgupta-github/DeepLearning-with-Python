# Importing the libraries and the dataset
import numpy as np
import datetime
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print(tf.__version__)

# Loading the dataset
# Dataset Source: https://www.kaggle.com/zalando-research/fashionmnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()


# Normalizing the images
# We divide each pixel of the image in the training and test sets by the maximum number of pixels (255).
#In this way each pixel will be in the range [0, 1]. By normalizing images we make sure that our model (ANN) trains faster.
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshaping
# Since each image's dimension is 28x28, we reshape the full dataset to [-1 (all elements), height * width]
X_train = X_train.reshape(-1, 28*28)
print(X_train.shape)
# We reshape the test set the same way
X_test = X_test.reshape(-1, 28*28)
print(X_test.shape)

# Building ANN
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=128, activation='relu', input_shape=(784, )))
# Dropout is a Regularization technique where we randomly set neurons in a layer to zero.
# That way while training those neurons won't be updated.
# Because some percentage of neurons won't be updated the whole training process is long and we have less chance for overfitting.
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Dense(units=64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

# Compiling the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
print(model.summary())

# Training the model
model.fit(X_train, y_train, epochs=10)

# Model Evaluation and prediction
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test accuracy: {}".format(test_accuracy))

# Saving the model
model_json = model.to_json()
with open("fashion_model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("fashion_model.h5")