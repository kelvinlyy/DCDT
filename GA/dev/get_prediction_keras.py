import sys
import os
import pickle
import redis
import numpy as np

'''
Arguments:
[1]: DL framework in Keras backend
[2]: db flag of redis server
[3]: layer index of predictions to get
'''
keras_bk = sys.argv[1]
db_flag = sys.argv[2]
layer_idx = sys.argv[3]

os.environ['KERAS_BACKEND'] = keras_bk
os.environ['CUDA_VISIBLE_DEVICES']=""
import keras


r = redis.Redis(db=db_flag)

# load models and inputs
model = pickle.loads(r.get("model"))
inputs = pickle.loads(r.get("inputs"))
    
# ensure inputs is 4-dimensional
if len(inputs.shape) == 3:
    inputs = inputs[None,:]

# collect model intermediate layers output function
layers_output = []
for l in model.layers:
    layers_output.append(l.output)
    
# predict
extractor = keras.Model(inputs=model.inputs, outputs=layers_output)
errors = []
try:
    outputs = extractor.predict(inputs)
except Exception as e:
    errors = [e]
else:
    for i in outputs:
        if np.isnan(i).any():
            errors = ['NaN']
            break

# save predictions
if layer_idx == 'all':
    r.hset("predictions", keras_bk, pickle.dumps(outputs))
elif layer_idx == 'error':
    pass
else:
    r.hset("predictions", keras_bk, outputs[int(layer_idx)].dumps())

# save errors
r.hset("errors", keras_bk, pickle.dumps(errors))


'''
Example usage:
python get_predicts.py "theano" 0 -1
'''