# This file provides dataset-specific parameters and functions for MNIST
# and CIFAR10.

import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet

def make_dataset(images, labels):
    return DataSet(images, labels, reshape=False, dtype=tf.uint8)


# MNIST

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist
import mnist_model as mnist_model
from util import save_mnist_images as mnist_save_images

MNIST_NUM_CLASSES = 10
MNIST_IMAGE_SIZE = 28

def mnist_example_shape(batch_size):
    return (batch_size, MNIST_IMAGE_SIZE * MNIST_IMAGE_SIZE)

def mnist_load_data():
    data_sets = input_data.read_data_sets('data')
    return data_sets.train, data_sets.validation, data_sets.test


# Only MNIST is available
def choose_dataset(set_name):
    if set_name.lower() == 'mnist':
        return mnist_model, mnist_save_images, MNIST_NUM_CLASSES, \
            MNIST_IMAGE_SIZE, mnist_example_shape, mnist_load_data
    else:
        return None
