import h5py
import os


# FedAvg for h5 files. (Not used in this project)
class Aggregator:
    def __init__(self,type='average'):
        self.type=type
    def initialize_weight(model,totalNumberOfModels):
        model_weights={}
        for i in model['model_weights'].keys():
            model_weights[i] = {}
            for j in model['model_weights'][i].keys():
                model_weights[i][j] = {}
                for z in model['model_weights'][i][j].keys():
                    model_weights[i][j][z] = []
                    for weight in model['model_weights'][i][j][z]:
                        model_weights[i][j][z].append(weight/totalNumberOfModels)
        return model_weights
    def add_weight(weights,model,totalNumberOfModels):
        for i in model['model_weights'].keys():
            for j in model['model_weights'][i].keys():
                for z in model['model_weights'][i][j].keys():
                    index = 0
                    for weight in model['model_weights'][i][j][z]:
                        weights[i][j][z][index] += weight/totalNumberOfModels
                        index += 1

    def insert_weight(weights,model):
        for i in model['model_weights'].keys():
            for j in model['model_weights'][i].keys():
                for z in model['model_weights'][i][j].keys():
                    index = 0
                    for weight in weights[i][j][z]:
                        model['model_weights'][i][j][z][index] = weight
                        index +=1
    def federated_average(self):
        models_to_averaged = list(filter(lambda x : x.startswith('lenet'), os.listdir()))
        main_model = h5py.File(models_to_averaged[0],"r+")
        model_weights = self.initialize_weight(main_model,len(models_to_averaged))
        for model_name in models_to_averaged[1:]:
            temp_model = h5py.File(model_name,"r")
            self.add_weight(model_weights,temp_model,len(models_to_averaged))
            temp_model.close()
            os.remove(model_name)
        print('Mean of weights is calculated !')
        self.insert_weight(model_weights,main_model)
        main_model.close()
        os.rename(models_to_averaged[0],"lenet_global.h5")
        print('Model is ready to be sent!')
