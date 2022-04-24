import sys
import os
import pickle
import redis
import numpy as np
import coverage


def calc_coverage(model, test_data):
    cov = coverage.Coverage()
    cov.start()
    
    predictions = model.predict(test_data)
    cov.stop()    
    covData = cov.get_data()
    covList = [(files, sorted(covData.lines(files))) for files in covData.measured_files()]      
    
    return covList
 

def total_lines(covList):
    cnt = 0
    for _ in covList:
        cnt += len(_[1])
    return cnt

'''
Arguments:
[1]: DL framework in Keras backend
[2]: db flag of redis server
[3]: model key
[4]: input key
[5]: coverage key
'''

keras_bk = sys.argv[1]
db_flag = sys.argv[2]
model_key = sys.argv[3]
input_key = sys.argv[4]
coverage_key = sys.argv[5]

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
total_lines = total_lines(calc_coverage(model, x))
r.hset(f"coverage_{coverage_key}", keras_bk, pickle.dumps(total_lines))

'''
Example usage:
python get_coverage_keras.py tensorflow 1 0 0 0
'''