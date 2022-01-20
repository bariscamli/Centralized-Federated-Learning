from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AveragePooling2D
import pandas as pd
from tensorflow.python.keras.utils.np_utils import to_categorical
import tensorflow as tf
import os
import numpy as np
import random


def get_data(client,data='cifar_10',iid="iid",balanced='balanced'):
    """
    Reads csv files and prepares datasets for model training & testing.

    :param client: Client ID
    :param data: Name of dataset
    :param iid: iid or non_iid
    :param balanced: balanced or unbalanced
    :return: Train and test datasets
    """

    # initialized for Cifar-10
    pixel = 32
    rgb = 3

    if data == 'mnist':
        pixel = 28
        rgb = 1

    if client=='server':
        # Server does not train model
        X_train=None
        y_train = None
    else:
        X_train = pd.read_csv(f'data/{data}/{iid}_{balanced}/{client}_X_train.csv',index_col=0)
        y_train = pd.read_csv(f'data/{data}/{iid}_{balanced}/{client}_y_train.csv',index_col=0)
        X_train = X_train / 255.0
        X_train = X_train.values.reshape(X_train.shape[0], pixel, pixel, rgb)
        y_train = y_train['label'].values
        y_train = to_categorical(y_train, num_classes=10)


    X_test = pd.read_csv(f'data/{data}/test.csv', index_col=0)
    y_test = X_test['label']
    X_test.drop('label', inplace=True, axis=1)
    X_test = X_test / 255.0
    X_test = X_test.values.reshape(X_test.shape[0], pixel, pixel, rgb)
    y_test = y_test.values
    return X_train,y_train,X_test,y_test


def get_model(data_name):
    """
    Creates LeNet-5 model.

    :param data_name: Name of dataset
    :return: Keras model
    """

    # initialized for Cifar-10
    pixel = 32
    rgb = 3

    if data_name == 'mnist':
        pixel=28
        rgb=1

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    SEED = 1
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # LeNet-5
    model = Sequential()
    model.add(Conv2D(6, kernel_size=5, strides=1,  activation='tanh',input_shape=(pixel, pixel, rgb)))
    model.add(AveragePooling2D())
    model.add(Conv2D(16, kernel_size=5, strides=1,  activation='tanh'))
    model.add(AveragePooling2D())
    model.add(Flatten())
    model.add(Dense(120, activation='tanh'))
    model.add(Dense(84, activation='tanh'))
    model.add(Dense(10, activation='softmax'))
    return model


