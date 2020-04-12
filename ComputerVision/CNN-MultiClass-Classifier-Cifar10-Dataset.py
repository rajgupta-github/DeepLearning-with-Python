import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("Tensorflow version used:", tf.__version__)

# Setting class names in the dataset
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Loading the dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print("Output Shape", y_train.shape)


#Image Normalization
X_train = X_train / 255.0
X_test = X_test / 255.0

# To see an image
# plt.imshow(X_test[10])

print("X Train Shape", X_train.shape)

# building the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=[32, 32, 3]))
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))
print("Model Summary:", model.summary())

# Compiling the model
model.compile(loss="sparse_categorical_crossentropy", optimizer="Adam", metrics=["sparse_categorical_accuracy"])

# training the model
model.fit(X_train, y_train, epochs=5)

# Model Evaluation and Predictions
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test accuracy: {}".format(test_accuracy))