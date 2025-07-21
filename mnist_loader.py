import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras.layers import Dense
from keras import models, layers
import numpy as np
import matplotlib.pyplot as plt


# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Print information about the dataset
print(f"Training data shape: {x_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test data shape: {x_test.shape}")
print(f"Test labels shape: {y_test.shape}")
print(f"Number of classes: {len(set(y_train))}")
print(f"Pixel value range: {x_train.min()} to {x_train.max()}")
print(f"First 10 training labels: {y_train[:10]}")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print(x_train)
print(y_train)

# plt.imshow(x_train[0], cmap='gray')
# plt.title(f"Label: {y_train[0]}")
# plt.show()

print(x_train[0])

x_train = x_train / 255.0
x_test = x_test / 255.0

print(x_train[0])
print(y_train[0])

model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")

predictions = model.predict(x_test)
print(f"Predictions shape: {predictions.shape}")
print(f"First prediction: {predictions[0]}")
print(f"First prediction: {np.argmax(predictions[0])}")

plt.imshow(x_test[0], cmap='gray')
plt.title(f"Predicted label: {np.argmax(predictions[0])}")
plt.show()