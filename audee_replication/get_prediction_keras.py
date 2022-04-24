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
'''

keras_bk = sys.argv[1]
db_flag = sys.argv[2]
model_key = sys.argv[3]
input_key = sys.argv[4]
predictions_key = sys.argv[5]

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

# do predictions
outputs = model.predict(x)
r.hset(f"predictions_{predictions_key}", keras_bk, pickle.dumps(outputs))


'''
Example usage:
python get_prediction_keras.py tensorflow 1 0 0 0
'''