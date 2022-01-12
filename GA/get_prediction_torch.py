import sys
import os
import pickle
import redis
import torch

'''
Arguments:
[1]: db flag of redis server
[2]: layer index of predictions to get
'''
db_flag = sys.argv[1]
layer_idx = int(sys.argv[2])

r = redis.Redis(db=db_flag)

model = pickle.loads(r.get("model_torch"))
inputs = torch.from_numpy(pickle.loads(r.get("inputs")))

with torch.no_grad():
    outputs = model.double()(inputs.type(torch.float64))
    
r.hset('predictions', "torch", pickle.dumps(outputs.numpy()))

''' CPU
def hook(model, input, output):
    global inter_output
    inter_output = output.detach()
    
    
r = redis.Redis(db=db_flag)

# load models and inputs
model = pickle.loads(r.get("model_torch"))
inputs = torch.from_numpy(pickle.loads(r.get("inputs")))

# make sure inputs are in (num of imgs, 3, height, width) format
if inputs.shape[1] != 3: # RGB channel first
    inputs = inputs.permute((0, 3, 1, 2))

# Using CPU
inputs = inputs.type(torch.FloatTensor)
model = model.to('cpu')

# check if layer_idx is acceptable
layers = [module for module in model.modules() if not isinstance(module, nn.Sequential)]
if abs(layer_idx) > len(layers):
    raise Exception("Layer index out of range")

layers[layer_idx].register_forward_hook(hook)
model(inputs)

# save predictions
r.hset('predictions', "torch", pickle.dumps(inter_output.cpu().numpy()))
'''

'''
Using GPU
try:
    # predict
    with torch.cuda.device(1):
        inputs = inputs.type(torch.cuda.FloatTensor)
        model = model.to('cuda')

        # check if layer_idx is acceptable
        layers = [module for module in model.modules()]
        if abs(layer_idx) > len(layers):
            raise Exception("Layer index out of range")

        layers[layer_idx].register_forward_hook(hook)
        model(inputs)


    # save predictions
    r.hset('predictions', "torch", pickle.dumps(inter_output.cpu().numpy()))

except Exception as e:
    print("Error occured:", e)
'''    

'''
Example usage:
python get_predicts_tf.py 0 -1
'''