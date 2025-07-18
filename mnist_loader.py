import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras.layers import Dense


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