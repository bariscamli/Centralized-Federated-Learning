import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

def seperate_iid(numberOfClients=10):
    X = pd.read_csv('../data/digit-recognizer/train.csv')
    print(X.head())
    y = X['label']
    X.drop('label',inplace=True,axis=1)

    clients = [[] for i in range(10)]
    for i in range(10):
        print('Label: ',i,' Total num: ',len(list(np.array(y[y ==i]))))
        np.random.shuffle(np.array(y[y ==i].index))
        temp_array= np.array_split(np.random.shuffle(np.array(y[y == i].index)), 10)
        for j in range(10):
            clients[j].append(list(temp_array[i]))






if __name__ == "__main__":
    seperate_iid()