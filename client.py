import socketio
from tensorflow.keras.models import model_from_json
from utils.train import get_data
from utils.model_utils import encode_layer,decode
import sys
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import confusion_matrix

class Node:
    def __init__(self, address, client, epochs=10):
        self.client = client
        self.server = address
        self.sio = socketio.Client()
        self.register_handles()

        self.X_train, self.y_train, self.X_test, self.y_test = get_data(self.client,'cifar_10','non_iid')
        self.model = None
        self.epochs = epochs

    def connect(self):
        self.sio.connect(url=self.server)

    def register_handles(self):
        self.sio.on("connection_received", self.connection_received)
        self.sio.on("start_training", self.start_training)

    def connection_received(self):
        print(f"Server at {self.server} returned success")

    def start_training(self, global_model):
        self.model = model_from_json(global_model["model_architecture"])
        self.model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.set_weights(decode(global_model["model_weights"]))
        print("Starting training")
        self.model.fit(self.X_train,self.y_train,epochs=self.epochs,batch_size=10)
        y_pred = self.model.predict(self.X_test)
        print('Accuracy: ',accuracy_score(self.y_test,np.argmax(y_pred, axis=1)))
        #print(confusion_matrix(self.y_test,  np.argmax(y_pred, axis=1)))
        self.send_updates()

    def send_updates(self):
        model_weights = dict()
        for layer in self.model.layers:
            if layer.trainable_weights:
                model_weights[layer.name] = encode_layer(layer.get_weights())

        self.sio.emit("fl_update",data=model_weights)

    def disconnect(self):
        return

    def end_session(self, data):
        model_weights = decode(data['model_weights'])
        self.model.set_weights(model_weights)


if __name__ == "__main__":
    node = Node(address="http://0.0.0.0:5000", client="client"+str(sys.argv[1]), epochs=10)
    node.connect()
