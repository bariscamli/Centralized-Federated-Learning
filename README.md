# Centralized Federated Learning

_Centralized Federated Learning using WebSockets and TensorFlow. Lenet is preferred for computer vision model due to simplicity. CIFAR-10 and MNIST are used as dataset._


- Install the dependencies:

```
pip install -r requirements.txt
```

- Run a server 
```
python server.py <number_of_requested_nodes> <number_of_communication_round> <name_of_dataset> <iid_or_non_iid> <balanced_or_unbalanced>
```

- Run a client
```
python client.py <client_id> <number_of_local_epoch>
```

## References
* <div>WebSocket implementation adopted by <a href="https://github.com/DhanshreeA/europython-minimal-fl" title="europython-minimal-fl">europython-minimal-fl</a></div>
* <div>Aggregation method (FedAvg) taken from <a href="https://arxiv.org/pdf/1602.05629.pdf" title="Communication-Efficient Learning of Deep Networks
from Decentralized Data">Communication-Efficient Learning of Deep Networks
from Decentralized Data</a></div>
* <div>MNIST Dataset taken from <a href="https://www.tensorflow.org/datasets/catalog/mnist" title="MNIST">MNIST</a></div>
* <div>CIFAR-10 Dataset taken from <a href="https://www.tensorflow.org/datasets/catalog/cifar10" title="CIFAR-10">CIFAR10</a></div>








