# Centralized Federated Learning

_Federated learning is a distributed machine learning technique to train neural network models across edge devices with small amount of data; instead, training model in cloud or server with centralized, big data corpus. In this project, I implemented an application that clients can interact with a server to be part of federated learning. Server is responsible for receiving, sending, and aggregating neural network models. The clients are only assigned to training corresponding model with their local data. I used Centralized Federated Learning using WebSockets and TensorFlow. Lenet is preferred for computer vision model due to simplicity. CIFAR-10 and MNIST are used as dataset._

<div>More details about implementation and test results can be found in the <a href="https://github.com/bariscamli/federated-learning/blob/main/Centralized%20Federated%20Learning.pdf" title="Report">Project Report</a></div>


![Centralized Federated Learning](https://github.com/bariscamli/federated-learning/blob/main/images/image.png?raw=true)

## Setup

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

## Demo
- Server

![](images/server.gif)

- Client

![](images/client.gif)

## References
* <div>WebSocket implementation adopted by <a href="https://github.com/DhanshreeA/europython-minimal-fl" title="europython-minimal-fl">europython-minimal-fl</a></div>
* <div>Aggregation method (FedAvg) taken from <a href="https://arxiv.org/pdf/1602.05629.pdf" title="Article">Communication Efficient Learning of Deep Networks
from Decentralized Data</a></div>
* <div>MNIST Dataset taken from <a href="https://www.tensorflow.org/datasets/catalog/mnist" title="MNIST">MNIST</a></div>
* <div>CIFAR-10 Dataset taken from <a href="https://www.tensorflow.org/datasets/catalog/cifar10" title="CIFAR-10">CIFAR10</a></div>

## License

This project is licensed under the terms of the Baris Camli License.









