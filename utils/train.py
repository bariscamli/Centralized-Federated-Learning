from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AveragePooling2D, MaxPool2D,MaxPooling2D
import pandas as pd
from tensorflow.python.keras.utils.np_utils import to_categorical
import tensorflow as tf
import os
import numpy as np
import random


def get_data(client,data='cifar_10',iid="iid"):
    pixel = 32
    rgb = 3
    if data == 'mnist':
        pixel = 28
        rgb = 1
    if client=='server':
        X_train=None
        y_train = None
    else:
        X_train = pd.read_csv(f'data/{data}/{iid}/{client}_X_train.csv',index_col=0)
        y_train = pd.read_csv(f'data/{data}/{iid}/{client}_y_train.csv',index_col=0)
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


def get_model_mnist():
    model = Sequential()
    model.add(Conv2D(filters=6, kernel_size=(3, 3), activation='tanh', input_shape=(28,28,1)))
    model.add(AveragePooling2D())
    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='tanh'))
    model.add(AveragePooling2D())
    model.add(Flatten())
    model.add(Dense(units=128, activation='tanh'))
    model.add(Dense(units=10, activation = 'softmax'))
    return model

def get_model_cifar():
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    SEED = 1
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    model = Sequential()
    # 30 Conv Layer
    model.add(Conv2D(30, kernel_size=(3, 3), padding='valid', activation='relu', input_shape=(32, 32, 3)))
    # 15 Max Pool Layer
    model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))
    # 13 Conv Layer
    model.add(Conv2D(13, kernel_size=(3,3), padding='valid', activation='relu'))
    # 6 Max Pool Layer
    model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))
    # Flatten the Layer for transitioning to the Fully Connected Layers
    model.add(Flatten())
    # 120 Fully Connected Layer
    model.add(Dense(120, activation='relu'))
    # 86 Fully Connected Layer
    model.add(Dense(86, activation='relu'))
    # 10 Output
    model.add(Dense(10, activation='softmax'))
    return model
