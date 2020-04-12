import tensorflow as tf
from tensorflow.keras.datasets import imdb
import numpy as np

print("tensorflow version", tf.__version__)

number_of_words = 20000
max_len = 100

# Loading the dataset
np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=number_of_words)
np.load = np_load_old

# Padding all sequences to be the same length
X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_len)
X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_len)

# Building the RNN
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=number_of_words, output_dim=128, input_shape=(X_train.shape[1],)))
model.add(tf.keras.layers.LSTM(units=128, activation='tanh'))
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Comopiling the model
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

print("Model Summary", model.summary())

# Training the model
model.fit(X_train, y_train, epochs=3, batch_size=128)

# Model Evaluation
test_loss, test_acurracy = model.evaluate(X_test, y_test)
print("Test accuracy: {}".format(test_acurracy))