import codecs
import pickle

def decode(b64_str):
    return pickle.loads(codecs.decode(b64_str.encode(), "base64"))

def encode_layer(layer):
    return codecs.encode(pickle.dumps(layer), "base64").decode()


