import sys
import os
import numpy as np

'''
Arguments:
[1]: DL framework in Keras backend
[2]: DNN model path
[3]: Path of the input set to be parsed into the DNN model
[4]: Path of the predictions to be saved to
'''
keras_bk = sys.argv[1]
model_path = sys.argv[2]
input_path = sys.argv[3]
save_path = sys.argv[4]

os.environ['KERAS_BACKEND'] = keras_bk

import keras

model = keras.models.load_model(model_path, compile=False)

inputs = np.load(input_path)

outputs = model.predict(inputs)

np.save(save_path, outputs)