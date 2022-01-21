import asyncio
import socketio

from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import accuracy_score
from aiohttp import web
import numpy as np

from utils.model_utils import encode_layer, decode
from utils.training import get_model, get_data


class Server:
    def __init__(self, req_nodes, comunication_rounds):
        self.sio = socketio.AsyncServer(async_mode="aiohttp", ping_timeout=120)
        self.app = web.Application()
        self.sio.attach(self.app)
        self.register_handles()

        self.connected_nodes = list()
        self.pending_nodes = list()
        self.req_nodes = req_nodes
        self.training_room = "training_room"
        self.average_weights = dict()

        _, _, self.X_test, self.y_test = get_data('server', 'mnist', 'non_iid', 'unbalanced')
        self.global_model = get_model("mnist")
        self.max_rounds = comunication_rounds
        self.round = 0
        self.pool = ThreadPoolExecutor(max_workers=4)

    def register_handles(self):
        self.sio.on("connect", self.connect)
        self.sio.on("fl_update", self.fl_update)

    async def connect(self, sid, environ):
        self.connected_nodes.append(sid)
        self.sio.enter_room(sid, self.training_room)

        async def start_training_callback():
            if len(self.connected_nodes) == self.req_nodes:
                print("Connected to ", self.req_nodes, " starting training")
                await self.start_round()
            else:
                print("Waiting to connect to ", self.req_nodes - len(self.connected_nodes),
                      " more nodes to start training")

        await self.sio.emit(
            "connection_received",
            room=sid,
            callback=start_training_callback,
        )

    def run_server(self, host="0.0.0.0", port=5000):
        web.run_app(self.app, host=host, port=port)

    def evaluate(self):
        y_pred = self.global_model.predict(self.X_test)
        print('Accuracy: ', accuracy_score(self.y_test, np.argmax(y_pred, axis=1)))

    async def fl_update(self, sid, data):
        for layer in data.keys():
            temp_weight = decode(data[layer])
            if len(self.average_weights[layer]) == 2:
                self.average_weights[layer][0] += (temp_weight[0] / len(self.connected_nodes))
                self.average_weights[layer][1] += (temp_weight[1] / len(self.connected_nodes))
            else:
                self.average_weights[layer].append(temp_weight[0] / len(self.connected_nodes))
                self.average_weights[layer].append(temp_weight[1] / len(self.connected_nodes))
        self.pending_nodes.remove(sid)
        if not self.pending_nodes:
            loop = asyncio.get_event_loop()
            asyncio.ensure_future(self.async_consume(loop))

    def apply_updates(self):
        print("Applying updates to global model")
        for layer in self.global_model.layers:
            if layer.trainable_weights:
                layer.set_weights(self.average_weights[layer.name])
        self.evaluate()

    def async_consume(self, loop):
        yield from loop.run_in_executor(self.pool, self.apply_updates)
        loop.create_task(self.end_round())

    async def start_round(self):
        print(f'Starting round {self.round + 1}')
        self.pending_nodes = self.connected_nodes.copy()
        for layer in self.global_model.layers:
            if layer.trainable_weights:
                self.average_weights[layer.name] = []
        await self.sio.emit(
            "start_training",
            data={
                "model_architecture": self.global_model.to_json(),
                "model_weights": encode_layer(self.global_model.get_weights()),
            },
            room=self.training_room,
        )

    async def end_round(self):
        print("Ending round")
        self.round += 1
        if self.round < self.max_rounds:
            await self.start_round()
        else:
            await self.end_session()

    async def end_session(self):
        print("Ending session")
        await self.sio.emit(
            "end_session",
            room=self.training_room,
            data={
                "model_weights": encode_layer(self.global_model.get_weights()),
            },
        )

    async def disconnect(self, sid):
        self.connected_nodes.remove(sid)
        self.sio.leave_room(sid, room=self.training_room)


if __name__ == "__main__":
    fl_server = Server(req_nodes=2, comunication_rounds=1)
    fl_server.run_server()
