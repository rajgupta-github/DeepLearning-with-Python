# https://github.com/GoogleCloudPlatform/keras-idiomatic-programmer/blob/master/books/deep-learning-design-patterns/Workshops/Novice/Deep%20Learning%20Design%20Patterns%20-%20Workshop%20-%20Chapter%20II.ipynb
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.datasets import cifar10
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



# Basic CNN as Sequential API
model = Sequential()
input_shape=(32, 32, 3)
model.add(Conv2D(filters=16, kernel_size=3, strides=2, activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=2, strides=2))
model.add(Conv2D(filters=32, kernel_size=3, strides=2, activation='relu'))
model.add(MaxPooling2D(pool_size=2, strides=2))
model.add(Dense(10, activation='softmax'))
print("Model Summary", model.summary())



# VGG16 as Sequential API
def conv_block(n_layers, n_filters):
    """
        n_layers : number of convolutional layers
        n_filters: number of filters
    """
    for n in range(n_layers):
        model.add(Conv2D(n_filters, (3, 3), strides=(1, 1), padding="same",
                  activation="relu"))
    model.add(MaxPooling2D(2, strides=2))

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same', activation="relu",input_shape=(224, 224, 3)))
conv_block(1, 64)
conv_block(2, 128)
conv_block(3, 256)
conv_block(3, 512)
conv_block(3, 512)
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dense(4096, activation='relu'))
model.add(Dense(1000, activation='softmax'))
print("VGG16 Model Summary", model.summary())


# 6 layer VGG
def makeVGG6():
    def conv_block(n_layers, n_filters):
        """
            n_layers : number of convolutional layers
            n_filters: number of filters
        """
        for n in range(n_layers):
            model.add(Conv2D(n_filters, (3, 3), strides=(1, 1), padding="same",
                             activation="relu"))
        model.add(MaxPooling2D(2, strides=2))

    model = Sequential()
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation="relu",
                     input_shape=(32, 32, 3)))

    # These are the convolutional groups
    conv_block(1, 64)
    conv_block(2, 128)
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
    return model
vgg6 = makeVGG6()
print("VGG6 Model Summary", vgg6.summary())


# 10 Layer VGG
def makeVGG10():
    def conv_block(n_layers, n_filters):
        """
            n_layers : number of convolutional layers
            n_filters: number of filters
        """
        for n in range(n_layers):
            model.add(Conv2D(n_filters, (3, 3), strides=(1, 1), padding="same",
                             activation="relu"))
        model.add(MaxPooling2D(2, strides=2))

    model = Sequential()
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation="relu",
                     input_shape=(32, 32, 3)))

    # These are the convolutional groups
    conv_block(1, 64)
    conv_block(2, 128)
    conv_block(3, 256)
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
    return model
vgg10 = makeVGG10()
print("VGG10 Model Summary", vgg10.summary())



# Let's get the tf.Keras builtin dataset for CIFAR-10.
# These are 32x32 color images (3 channels) of 10 classes (airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks).
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = (x_train / 255.0).astype(np.float32)

# Let's train both the 6 and 10 layer VGG on CIFAR-10 for 3 epochs and compare the results.
vgg6.fit(x_train, y_train, epochs=3, batch_size=32, validation_split=0.1, verbose=1)
vgg10.fit(x_train, y_train, epochs=3, batch_size=32, validation_split=0.1, verbose=1)
