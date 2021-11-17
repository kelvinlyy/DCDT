import sys
import os
import pickle
import redis

'''
Arguments:
[1]: DL framework in Keras backend
[2]: db flag of redis server
[3]: layer index of predictions to get
'''
keras_bk = sys.argv[1]
db_flag = sys.argv[2]
layer_idx = sys.argv[3]

# import keras with backend=keras_bk
os.environ['KERAS_BACKEND'] = keras_bk
os.environ['CUDA_VISIBLE_DEVICES']=""
import keras


r = redis.Redis(db=db_flag)

# load models and inputs
model = pickle.loads(r.get("model"))
inputs = pickle.loads(r.get("inputs"))

# check if layer_idx is acceptable
if abs(int(layer_idx)) > len(model.layers):
    raise Exception("Layer index out of range")

# predict
extractor = keras.Model(inputs=model.inputs, outputs=model.layers[int(layer_idx)].output)
outputs = extractor.predict(inputs)

# save predictions
r.hset(keras_bk, "predictions", outputs.dumps())


'''
Example usage:
python get_predicts.py "theano" 0 -1
'''