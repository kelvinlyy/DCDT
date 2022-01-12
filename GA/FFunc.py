import redis
import os
import numpy as np
import onnx
import onnx_tf
from onnx2keras import onnx_to_keras
import pickle
import tensorflow as tf
import torch
import keras
from multiprocessing import Process



def export_model(model_framework, model, model_name, dummy_input, onnx_path):
    if model_framework == "tensorflow":
        model.save(model_name)
        model_path = model_name
        os.system(f"python -m tf2onnx.convert --saved-model {model_path} --output {onnx_path}")
    elif model_framework == "torch":
        torch.onnx.export(model, dummy_input.to('cuda'), onnx_path)


def import_onnx(target_framework, onnx_path): # call this function after export_tf_model()
    onnx_model = onnx.load(onnx_path)  # load onnx model
    if target_framework == 'tensorflow':
        output_path = onnx_path[:-5]
        with tf.device('CPU'):
            tf_rep = onnx_tf.backend.prepare(onnx_model)  # prepare tf representation
            tf_rep.export_graph(output_path)  # export the model
    elif target_framework == 'theano':
        output_path = onnx_path[:-5] + '.h5'
        # Call the converter (input - is the main model input name, can be different for your model)
        k_model = onnx_to_keras(onnx_model, ['input'])
        k_model.save(output_path)
        
    
    return output_path

def load_model(model_framework, model_path):
    if model_framework == "tensorflow":
        with tf.device('CPU'):
            model = tf.keras.models.load_model(model_path)
    elif model_framework == "theano":
        model = keras.models.load_model(model_path)
       
    return model


def get_prediction_process(model_framework, db_flag, layer_idx):
    if model_framework == "theano":
        cmd = f"/data/yylaiai/anaconda3/envs/py36/bin/python get_prediction_{model_framework}.py {db_flag} {layer_idx}"
    else:
        cmd = f"/data/yylaiai/anaconda3/envs/fyp_v1/bin/python get_prediction_{model_framework}.py {db_flag} {layer_idx}"

    return Process(target=lambda: os.system(cmd))

class InconsistencyFFunc:
    def __init__(self, redis_server, db_flag, backends, model, inputs):
        self.redis_server = redis_server
        self.db_flag = db_flag
        self.backend_1, self.backend_2 = backends
        self.model_1 = model
        self.inputs = inputs

        self.model_1_name = 'model' + self.backend_1
        self.onnx_path = self.model_1_name + '.onnx'

    def prepare(self):
        # export model to onnx
        export_model(self.backend_1, self.model_1, self.model_1_name, self.inputs, self.onnx_path)

        # import onnx model to second DL framework
        self.model_2_path = import_onnx(self.backend_2, self.onnx_path)
        self.model_2 = load_model(self.backend_2, self.model_2_path)

        # store model and inputs
        if self.backend_1 == "torch":
            with self.redis_server.pipeline() as pipe:
                pipe.mset({f'model_{self.backend_1}': pickle.dumps(self.model_1.to('cpu'))})
                pipe.mset({f'model_{self.backend_2}': pickle.dumps(self.model_2)})
                pipe.mset({"inputs": pickle.dumps(self.inputs.cpu().numpy())}) # convert tensor to numpy before saving to redis
                pipe.execute()
        else:
            with self.redis_server.pipeline() as pipe:
                pipe.mset({f'model_{self.backend_1}': pickle.dumps(self.model_1)})
                pipe.mset({f'model_{self.backend_2}': pickle.dumps(self.model_2)})
                pipe.mset({"inputs": pickle.dumps(self.inputs)})
                pipe.execute()
    
    def compute(self):
        # run subprocess to get predictions
        p1 = get_prediction_process(self.backend_1, self.db_flag, layer_idx)
        p2 = get_prediction_process(self.backend_2, self.db_flag, layer_idx)
        p1.start()
        p2.start()
        p1.join()
        p2.join()
        
        # load predictions
        with self.redis_server.pipeline() as pipe:
            pipe.hget("predictions", self.backend_1)
            pipe.hget("predictions", self.backend_2)
            predictions = pipe.execute()
        
        self.predictions_1 = pickle.loads(predictions[0])
        self.predictions_2 = pickle.loads(predictions[1])
        
        # compute fitness
        predictions_diff = np.abs(self.predictions_2 - self.predictions_1)
        
        self.fitness_values = np.sum(predictions_diff**2, axis=1) / len(self.predictions_1)
        
        return self.fitness_values

class NanFFunc:
    def __init__(self, redis_server, db_flag, backend, model, inputs):
        self.redis_server = redis_server
        self.db_flag = db_flag
        self.backend = backend
        self.model = model
        self.inputs = inputs


    def prepare(self):
        # store model and inputs
        if self.backend == "torch":
            with self.redis_server.pipeline() as pipe:
                pipe.mset({f'model_{self.backend}': pickle.dumps(self.model)})
                pipe.mset({"inputs": pickle.dumps(self.inputs.cpu().numpy())}) # convert tensor to numpy before saving to redis
                pipe.execute()
        else:
            with self.redis_server.pipeline() as pipe:
                pipe.mset({f'model_{self.backend}': pickle.dumps(self.model)})
                pipe.mset({"inputs": pickle.dumps(self.inputs)})
                pipe.execute()
            
    
    def compute(self, layer_idx):
        # run subprocess to get predictions
        p = get_prediction_process(self.backend, self.db_flag, layer_idx)
        p.start()
        p.join()
        
        # load predictions
        self.predictions = pickle.loads(self.redis_server.hget("predictions", self.backend))

        # normalize neurons
        normalized_outputs = self.predictions/(np.amax(self.predictions) - np.amin(self.predictions))
        
        # compute fitness
        self.fitness_values = np.amax(normalized_outputs, axis=1) - np.amin(normalized_outputs, axis=1)
        
        return self.fitness_values


