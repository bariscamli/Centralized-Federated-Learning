from sklearn.metrics import accuracy_score

from utils.train import get_model_cifar, get_data, get_model_mnist
import pandas as pd
import numpy as np
from tensorflow.python.keras.utils.np_utils import to_categorical


data = 'cifar_10'
X_train = pd.read_csv(f'data/{data}/train.csv', index_col=0)
y_train = X_train['label']
X_train.drop('label',inplace=True,axis=1)
X_train = X_train / 255.0
X_train = X_train.values.reshape(X_train.shape[0], 32, 32, 3)

y_train = y_train.values
y_train = to_categorical(y_train, num_classes=10)

X_test = pd.read_csv(f'data/{data}/test.csv', index_col=0)
y_test = X_test['label']
X_test.drop('label', inplace=True, axis=1)
X_test = X_test / 255.0
X_test = X_test.values.reshape(X_test.shape[0], 32, 32, 3)
y_test = y_test.values

if data == 'cifar_10':
    model = get_model_cifar()
else:
    model = get_model_mnist()
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train,y_train, epochs=10,batch_size=10,validation_data=(X_test, to_categorical(y_test, num_classes=10)))
y_pred = model.predict(X_test)
print('Accuracy: ', accuracy_score(y_test, np.argmax(y_pred, axis=1)))
