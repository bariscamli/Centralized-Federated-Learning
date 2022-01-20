import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.python.keras.utils.np_utils import to_categorical

from utils.training import get_model, get_data


DATA = 'mnist'

if __name__ == "__main__":
    pixel = 32
    rgb = 3
    if DATA == 'mnist':
        pixel = 28
        rgb = 1

    X_train, y_train, X_test, y_test = get_data('all', 'mnist', 'non_iid', 'unbalanced')

    model= get_model(DATA)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train,y_train, epochs=1,batch_size=10,validation_data=(X_test, to_categorical(y_test, num_classes=10)))
    y_pred = model.predict(X_test)
    print('Accuracy: ', accuracy_score(y_test, np.argmax(y_pred, axis=1)))
