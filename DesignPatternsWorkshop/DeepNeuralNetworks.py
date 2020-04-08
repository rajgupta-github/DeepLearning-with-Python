import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import Flatten, Dense, Activation
from tensorflow.keras.datasets import mnist

print(np.__version__)
print(tf.__version__)


def makeModelUsingSequentialLongForm():
    # Basic DNN as Sequential API (long form)
    input_shape = ((28, 28, 1))
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(128))
    model.add(Activation("relu"))
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(Dense(10))
    model.add(Activation("softmax"))
    print(model.summary())
    return model

def makeModelUsingSequentialShortForm():
    # Basic DNN as Sequential API (short form)
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28, 1)))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(512, activation="relu"))
    model.add(Dense(10, activation="softmax"))
    print(model.summary())
    return model

def makeModelUsingSequentialListForm():
    # Basic DNN as Sequential API (list form)
    model = Sequential([
        Flatten(input_shape=(28, 28, 1)),
        Dense(128, activation='relu'),
        Dense(512, activation='relu'),
        Dense(10, activation='softmax'),
    ])
    print(model.summary())
    return model

def makeModelUsingFunctionalAPI():
    # Make a DNN model
    inputs = Input((28, 28))
    x = Flatten()(inputs)
    x = Dense(128, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    outputs = Dense(10, activation='softmax')(x)
    model = Model(inputs, outputs)
    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
    return model

# We make two copies of the model
model_a = makeModelUsingFunctionalAPI()
model_b = makeModelUsingFunctionalAPI()

# Load the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# model_a.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)

x_train = (x_train / 255.0).astype(np.float32)
model_b.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)

# We see Model b loss and accuracy have improved compared to model a
# Model a Accuracy and Loss is:
# # # loss: 0.1125 - acc: 0.9709 - val_loss: 0.1529 - val_acc: 0.9685
# Model b Accuracy and Loss is:
# loss: 0.0235 - acc: 0.9925 - val_loss: 0.0997 - val_acc: 0.9803