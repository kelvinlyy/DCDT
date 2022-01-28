from multiprocessing import Process
import os
import numpy as np
import pickle

py_env = "/data/yylaiai/anaconda3/envs/tf_th/bin/python"
get_prediction_py = "get_prediction_keras.py"

def get_prediction_cmd(model_framework, db_flag, layer_idx):
    cmd = f"{py_env} {get_prediction_py} {model_framework} {db_flag} {layer_idx}"
    return cmd
    

class InconsistencyFFunc:
    def __init__(self, redis_server, db_flag, backends, model, inputs):
        self.redis_server = redis_server
        self.db_flag = db_flag
        self.backend_1, self.backend_2 = backends
        self.model = model
        self.inputs = inputs

    def prepare(self):
        # store model and inputs
        with self.redis_server.pipeline() as pipe:
            pipe.mset({"model": pickle.dumps(self.model)})
            pipe.mset({"inputs": pickle.dumps(self.inputs)})
            pipe.execute()
    
    def compute(self, layer_idx):
        # run subprocess to get predictions
        cmd_1 = get_prediction_cmd(self.backend_1, self.db_flag, layer_idx)
        cmd_2 = get_prediction_cmd(self.backend_2, self.db_flag, layer_idx)

        p1 = Process(target=lambda: os.system(cmd_1))
        p2 = Process(target=lambda: os.system(cmd_2))
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
        self.fitness_values = np.sum(predictions_diff, axis=1) / len(predictions_diff[0])

        return self.fitness_values


class NanFFunc:
    def __init__(self, redis_server, db_flag, backend, model, inputs):
        self.redis_server = redis_server
        self.db_flag = db_flag
        self.backend = backend[0]
        self.model = model
        self.inputs = inputs

    def prepare(self):
        # store model and inputs
        with self.redis_server.pipeline() as pipe:
            pipe.mset({"model": pickle.dumps(self.model)})
            pipe.mset({"inputs": pickle.dumps(self.inputs)})
            pipe.execute()
    
    def compute(self, layer_idx):
        os.system(get_prediction_cmd(self.backend, self.db_flag, layer_idx))

        self.predictions = pickle.loads(self.redis_server.hget("predictions", self.backend))
        self.fitness_values = np.amax(self.predictions, axis=1) - np.amin(self.predictions, axis=1)
        
        return self.fitness_values