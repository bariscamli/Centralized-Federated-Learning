
'''
import h5py
import io

import base64
import simplejson as json
with open('lenet.h5', 'rb') as f:
    #h = h5py.File(io.BytesIO(f.read()), 'r')
    #h.close()

    array = base64.b64encode(f.read())
    #print(array)
    #print(str(array))

    with open("data.json", "w") as f:
        json.dump({'key':array},f)


with open("data.json","r") as f:
    p = json.load(f)
    #print(bytes((p['key'])))

    with open('lenet_one.h5', "wb") as f_one:
        f_one.write(base64.b64decode(p['key']))




;
with open("lenet_final.h5", "wb") as binary_file:
    # Write bytes to file
    binary_file.write(f.read())
'''
import codecs
import pickle
from train import get_model

def encode(model):
    try:
        model_weights = model.get_weights()
    except AttributeError:
        model_weights = model
    return codecs.encode(pickle.dumps(model_weights), "base64").decode()

def decode(b64_str):
    return pickle.loads(codecs.decode(b64_str.encode(), "base64"))

model = get_model()
model.compile("adam", "binary_crossentropy", metrics=["accuracy"])

for name in model.layers:
    if name.trainable_weights:
        print(name.name)
        array = decode(encode(name.get_weights()))
        print(len(array))
        print(type(array[0]))
        print(type(array[1]))





'''
encoded_model_layers = [encode(lyr) for lyr in [layer.get_weights() for layer in model.layers]]

print(encoded_model_layers)

for idx, layer in enumerate(encoded_model_layers):
    print(idx)
    print(layer)
'''
