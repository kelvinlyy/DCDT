import sys
import os
import pickle
import redis
import numpy as np

'''
Arguments:
[1]: DL framework in Keras backend
[2]: db flag of redis server
[3]: model key
[4]: input key
[5]: predictions key
[6]: layer index of predictions to get
'''

keras_bk = sys.argv[1]
db_flag = sys.argv[2]
model_key = sys.argv[3]
input_key = sys.argv[4]
predictions_key = sys.argv[5]
layer_idx = sys.argv[6]

os.environ['KERAS_BACKEND'] = keras_bk
os.environ['CUDA_VISIBLE_DEVICES']=""
import keras


r = redis.Redis(db=db_flag)

# load models and inputs
model = pickle.loads(r.hget("model", model_key))
x = pickle.loads(r.hget("input", input_key))
    
# ensure inputs is 4-dimensional
if len(x.shape) == 3:
    x = x[None,:]

# collect model intermediate layers output function
layers_output = []
for l in model.layers:
    layers_output.append(l.output)
    
# predict
errors = []
try:
    extractor = keras.Model(inputs=model.inputs, outputs=layers_output)
    outputs = extractor.predict(x)
except Exception as e:
    r.hset(f"errors_{predictions_key}", keras_bk, pickle.dumps([e]))
    sys.exit(e)
else:
    er_layers = []
    isnan = False
    for i in range(len(outputs)):
        if np.isnan(outputs[i]).any():
            isnan = True
            er_layers.append(i)
    
    if isnan:
        errors.append('nan')
        errors.append(er_layers)

# save predictions
if layer_idx == 'all':
    r.hset(f"predictions_{predictions_key}", keras_bk, pickle.dumps(outputs))
elif layer_idx == 'error':
    pass
else:
    r.hset(f"predictions_{predictions_key}", keras_bk, outputs[int(layer_idx)].dumps())

# save errors
r.hset(f"errors_{predictions_key}", keras_bk, pickle.dumps(errors))


'''
Example usage:
python get_predicts.py "theano" 0 -1
'''