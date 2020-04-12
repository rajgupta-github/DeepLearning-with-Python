import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# Helper libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
# Import of keras model and hidden layers for our convolutional network
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.layers import Dense, Flatten

# Sklearn
from sklearn.model_selection import train_test_split  # Helps with organizing data for training
from sklearn.metrics import confusion_matrix  # Helps present results as a confusion-matrix

print(tf.__version__)

# We need to get all the paths for the images to later load them
imagepaths = []
# Go through all the files and subdirectories inside a folder and save path to images inside list
for root, dirs, files in os.walk("/Users/rajkgupta/Downloads/leapGestRecog/", topdown=False):
    for name in files:
        path = os.path.join(root, name)
        if path.endswith("png"):  # We want only the images
            imagepaths.append(path)
print(len(imagepaths))  # If > 0, then a PNG image was loaded


# This function is used more for debugging and showing results later. It plots the image into the notebook
def plot_image(path):
    img = cv2.imread(path)  # Reads the image into a numpy.array
    img_cvt = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Converts into the corret colorspace (RGB)
    print(img_cvt.shape)  # Prints the shape of the image just to check
    plt.grid(False)  # Without grid so we can see better
    plt.imshow(img_cvt)  # Shows the image
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.title("Image " + path)


X = []  # Image data
y = []  # Labels
# Loops through imagepaths to load images and labels into arrays
for path in imagepaths:
    img = cv2.imread(path)  # Reads image and returns np.array
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Converts into the corret colorspace (GRAY)
    img = cv2.resize(img, (320, 120))  # Reduce image size so training can be faster
    X.append(img)
    # Processing label in image path
    category = path.split("/")[6]
    label = int(category.split("_")[0][1])  # We need to convert 10_down to 00_down, or else it crashes
    y.append(label)
# Turn X and y into np.array to speed up train_test_split
X = np.array(X, dtype="uint8")
X = X.reshape(len(imagepaths), 120, 320, 1)  # Needed to reshape so CNN knows it's different images
y = np.array(y)
print("Images loaded: ", len(X))
print("Labels loaded: ", len(y))
print(y[0], imagepaths[0])  # Debugging

ts = 0.3  # Percentage of images that we want to use for testing. The rest is used for training.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=42)

# Construction of model
model = Sequential()
model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(120, 320, 1)))
model.add(MaxPool2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Configures the model for training
# Optimization routine, which tells the computer how to adjust the parameter values to minimize the loss function.
# Loss function, which tells us how bad our predictions are.
# List of metrics to be evaluated by the model during training and testing.
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("Model Summary", model.summary())

# Trains the model for a given number of epochs (iterations on a dataset) and validates it.
model.fit(X_train, y_train, epochs=5, batch_size=64, verbose=2, validation_data=(X_test, y_test))

# Save entire model to a HDF5 file
model.save('handrecognition_model.h5')

test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy: {:2.2f}%'.format(test_acc * 100))

predictions = model.predict(X_test)  # Make predictions towards the test set

np.argmax(predictions[0]), y_test[0]  # If same, got it right

#
# # Function to plot images and labels for validation purposes
# def validate_9_images(predictions_array, true_label_array, img_array):
#     # Array for pretty printing and then figure size
#     class_names = ["down", "palm", "l", "fist", "fist_moved", "thumb", "index", "ok", "palm_moved", "c"]
#     plt.figure(figsize=(15, 5))
#
#     for i in range(1, 10):
#         # Just assigning variables
#         prediction = predictions_array[i]
#         true_label = true_label_array[i]
#         img = img_array[i]
#         img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#
#         # Plot in a good way
#         plt.subplot(3, 3, i)
#         plt.grid(False)
#         plt.xticks([])
#         plt.yticks([])
#         plt.imshow(img, cmap=plt.cm.binary)
#
#         predicted_label = np.argmax(prediction)  # Get index of the predicted label from prediction
#
#         # Change color of title based on good prediction or not
#         if predicted_label == true_label:
#             color = 'blue'
#         else:
#             color = 'red'
#
#         plt.xlabel("Predicted: {} {:2.0f}% (True: {})".format(class_names[predicted_label],
#                                                               100 * np.max(prediction),
#                                                               class_names[true_label]),
#                    color=color)
#     plt.show()
#
#
# validate_9_images(predictions, y_test, X_test)

y_pred = np.argmax(predictions, axis=1)  # Transform predictions into 1-D array with label number

# H = Horizontal
# V = Vertical

pd.DataFrame(confusion_matrix(y_test, y_pred),
             columns=["Predicted Thumb Down", "Predicted Palm (H)", "Predicted L", "Predicted Fist (H)",
                      "Predicted Fist (V)", "Predicted Thumbs up", "Predicted Index", "Predicted OK",
                      "Predicted Palm (V)", "Predicted C"],
             index=["Actual Thumb Down", "Actual Palm (H)", "Actual L", "Actual Fist (H)", "Actual Fist (V)",
                    "Actual Thumbs up", "Actual Index", "Actual OK", "Actual Palm (V)", "Actual C"])

# Conclusion

# Based on the results presented in the previous section, we can conclude that our algorithm successfully
# classifies different hand gestures images with enough confidence (>95%) based on a Deep Learning model.

# The accuracy of our model is directly influenced by a few aspects of our problem. The gestures presented are
# reasonably distinct, the images are clear and without background. Also, there is a reasonable quantity of images,
# which makes our model more robust. The drawback is that for different problems, we would probably need more data to
# stir the parameters of our model into a better direction. Moreover, a deep learning model is very hard to
# interpret, given it's abstractions. However, by using this approach it becomes much more easier to start working on
# the actual problem, since we don't have to account for feature engineering. This means that we don't need to
# pre-process the images with edge or blob detectors to extract the important features; the CNN does it for us. Also,
# it can be adapted to new problems relatively easily, with generally good performance.

# As mentioned, another approach to this problem would be to use feature engineering, such as binary thresholding (
# check area of the hand), circle detection and others to detect unique characteristics on the images. However,
# with our CNN approach, we don't have to worry about any of these
