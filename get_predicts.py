import sys
import os
import numpy as np

'''
Arguments:
[1]: DL framework in Keras backend
[2]: DNN model path
[3]: Layer index of DNN model to be tested
[4]: Path of the input set to be parsed into the DNN model
[5]: Path of the predictions to be saved to
'''
keras_bk = sys.argv[1]
model_path = sys.argv[2]
input_path = sys.argv[3]
save_path = sys.argv[4]
layer_idx = sys.argv[5]

os.environ['KERAS_BACKEND'] = keras_bk

import keras

model = keras.models.load_model(model_path, compile=False)

if abs(int(layer_idx)) > len(model.layers):
    raise Exception("Layer index out of range")
    
inputs = np.load(input_path)

extractor = keras.Model(inputs=model.inputs, outputs=model.layers[int(layer_idx)].output)
outputs = extractor.predict(inputs)

np.save(save_path, outputs)

'''
Example usage:
python get_predicts.py "theano" "my_model.h5" "my_input.npy" "my_result.npy"
'''