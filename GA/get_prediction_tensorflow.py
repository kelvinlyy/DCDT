import sys
import os
import pickle
import redis

os.environ['CUDA_VISIBLE_DEVICES']=""

import tensorflow as tf

'''
Arguments:
[1]: db flag of redis server
[2]: layer index of predictions to get
'''
db_flag = sys.argv[1]
layer_idx = sys.argv[2]


r = redis.Redis(db=db_flag)

# load models and inputs
model = pickle.loads(r.get("model_tensorflow"))
inputs = pickle.loads(r.get("inputs"))

# make sure inputs are in (num of imgs, height, width, 3) format
if inputs.shape[-1] != 3:
    inputs = np.transpose(inputs, (0, 2, 3, 1))
    
# check if layer_idx is acceptable
if abs(int(layer_idx)) > len(model.layers):
    raise Exception("Layer index out of range")

# predict
extractor = tf.keras.Model(inputs=model.input, outputs=model.layers[int(layer_idx)].output)
outputs = extractor.predict(inputs)

# save predictions
r.hset('predictions', "tensorflow", outputs.dumps())


'''
Example usage:
python get_predicts_tf.py 0 -1
'''