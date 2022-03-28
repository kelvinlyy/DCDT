import os
import imp
import sys
import pickle
import redis
import torch
import numpy as np
import tensorflow as tf
from onnx_pytorch import code_gen

os.environ['CUDA_VISIBLE_DEVICES']=""

def convert_tf2torch(model, model_name, layer_idx):
    m = tf.keras.Model(inputs=model.input, outputs=model.layers[layer_idx].output)
    model_path = os.path.join('./torch_model', model_name + '_' + str(layer_idx))
    m.save(model_path)
    
    # export model to onnx
    os.system(f"/data/yylaiai/anaconda3/envs/fyp_v3/bin/python -m tf2onnx.convert --saved-model {model_path} --output {model_path}.onnx")
    
    if not os.path.exists(f"./torch_model/layer_{layer_idx}"):
        os.mkdir(f"./torch_model/layer_{layer_idx}")
        
    # convert onnx to torch
    code_gen.gen(f"{model_path}.onnx", f"./torch_model/layer_{layer_idx}")
    
    m = imp.load_source('model', f'torch_model/layer_{layer_idx}/model.py')
    model_torch = m.Model()
    
    return model_torch
    


'''
Arguments:
[1]: db flag of redis server
[2]: model key
[3]: input key
[4]: layer idx
'''

db_flag = sys.argv[1]
model_key = sys.argv[2]
input_key = sys.argv[3]
layer_idx = int(sys.argv[4])

r = redis.Redis(db=db_flag)

# load models and inputs
model = pickle.loads(r.hget("model", model_key))
x = pickle.loads(r.hget("input", input_key))

# ensure inputs is 4-dimensional
if len(x.shape) == 3:
    x = x[None,:]
    
# convert tf to torch
model_torch = convert_tf2torch(model, 'tf_model', layer_idx)
x_torch = torch.from_numpy(x.astype(np.float32))

# do predictions
predictions = model_torch(x_torch)
r.hset(f"predictions_{layer_idx}", 'torch', pickle.dumps(predictions))


'''
Example usage:
python get_prediction_torch.py 1 0 0 0
'''