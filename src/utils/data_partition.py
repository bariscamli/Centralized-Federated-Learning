import pandas as pd
import numpy as np

DATA = ['mnist', 'cifar']
CLIENT_NUMBER = 10
DATA_PER_CLIENT = [50000/CLIENT_NUMBER, 6000/CLIENT_NUMBER]
DATA_INDEX = 0


def read_data():
    """
    Read data from corresponding path.

    :return: DataFrame Object from csv file
    """
    X = pd.read_csv(f'../data/{DATA[DATA_INDEX]}/train.csv', index_col=0)
    y = X['label']
    X.drop('label', inplace=True, axis=1)
    return X, y


def save_as_csv(clients, X, y, type):
    """
    Read data from corresponding path.

    :param clients: Index list of each client
    :param X: Data
    :param y: Label
    :param type: Type of partition
    :return: DataFrame Object from csv file
    """
    for client_id in range(CLIENT_NUMBER):
        print(len(clients[client_id]))
        y[clients[client_id]].reset_index(drop=True)\
            .to_csv(f'../data/{DATA[DATA_INDEX]}/{type}/client' + str(client_id + 1) + '_y_train.csv')
        X.iloc[clients[client_id]].reset_index(drop=True)\
            .to_csv(f'../data/{DATA[DATA_INDEX]}/{type}/client' + str(client_id + 1) + '_X_train.csv')


def iid_balanced():
    """
    Independent and identically distributed and balanced partition.
    Each client has same amount of data while having data from all labels without overlapping.

    :return: None
    """
    X, y = read_data()
    rand_array = np.array(y.index)
    np.random.shuffle(rand_array)
    clients = [[] for i in range(CLIENT_NUMBER)]
    for i in range(CLIENT_NUMBER):
        clients[i] = list(rand_array[i * DATA_PER_CLIENT[DATA_INDEX]:(i + 1) * DATA_PER_CLIENT[DATA_INDEX]])
    save_as_csv(clients, X, y, "iid_balanced")


def non_iid_balanced():
    """
    Non Independent and identically distributed and balanced partition.
    Each client has same amount of data while having data from only two labels without overlapping.

    :return: None
    """
    X, y = read_data()
    first = []
    second = []
    temp_array = np.array_split(y.sort_values(), CLIENT_NUMBER * 2)
    for i in range(len(temp_array)):
        if i % 2 == 0:
            first.append(temp_array[i])
        else:
            second.append(temp_array[i])

    index_first = 0
    index_second = 0
    clients = [[] for i in range(CLIENT_NUMBER)]

    for i in range(CLIENT_NUMBER):
        if i % 2 == 0:
            clients[i] = list(np.concatenate((first[index_first].index, first[index_first + 1].index), axis=0))
            index_first += 2
        else:
            clients[i] = list(np.concatenate((second[index_second].index, second[index_second + 1].index), axis=0))
            index_second += 2
    save_as_csv(clients, X, y, "non_iid_balanced")


def non_iid_unbalanced():
    """
    Non Independent and identically distributed and unbalanced partition.
    Each client has different amount of data while having data from only two labels without overlapping.
    Data amount is separated randomly.

    :return: None
    """
    X, y = read_data()
    temp_array = np.array_split(y.sort_values(), CLIENT_NUMBER)
    clients = [[] for i in range(CLIENT_NUMBER)]
    for i in range(0, CLIENT_NUMBER, 2):
        cut = np.random.randint(y.shape[0] / CLIENT_NUMBER, size=(1, 1))[0][0]
        cut2 = np.random.randint(y.shape[0] / CLIENT_NUMBER, size=(1, 1))[0][0]

        clients[i] = list(np.concatenate(((temp_array[i][:cut]).index, (temp_array[i + 1][:cut2]).index), axis=0))
        clients[i + 1] = list(np.concatenate(((temp_array[i][cut:]).index, (temp_array[i + 1][cut2:]).index), axis=0))
    save_as_csv(clients, X, y, "non_iid_unbalanced")



