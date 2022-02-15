from multiprocessing import Process
import os
import numpy as np
import pickle

py_env = "/data/yylaiai/anaconda3/envs/tf_th/bin/python"
get_prediction_py = "get_prediction_keras.py"

# parse arguments to generate command for predicting input
def get_prediction_cmd(model_framework, db_flag, model_key, input_key, predictions_key, layer_idx):
    cmd = f"{py_env} {get_prediction_py} {model_framework} {db_flag} {model_key} {input_key} {predictions_key} {layer_idx}"
    return cmd


class InconsistencyFFunc:
    def __init__(self, redis_server, db_flag, mut_level, backends, model, model_weights, inputs):
        self.redis_server = redis_server
        self.db_flag = db_flag

        self.mut_level = mut_level
        self.backend_1, self.backend_2 = backends
        self.model = model
        self.model_weights = model_weights
        self.inputs = inputs

    def prepare(self):
        if self.mut_level == 'i':
            self.n = len(self.inputs)
            with self.redis_server.pipeline() as pipe: # 1 model, n inputs
                pipe.hset('model', 0, pickle.dumps(self.model))
                for i in range(len(self.inputs)):
                    pipe.hset('input', i, pickle.dumps(self.inputs[i]))
                pipe.execute()
        elif self.mut_level == 'w':
            self.n = len(self.model_weights)
            with self.redis_server.pipeline() as pipe: # 1 input, n models
                pipe.hset('input', 0, pickle.dumps(self.inputs[0]))
                for i in range(len(self.model_weights)):
                    self.model.set_weights(self.model_weights[i])
                    pipe.hset('model', i, pickle.dumps(self.model))
                pipe.execute()
        elif self.mut_level == 'i+w':
            assert len(self.model_weights) == len(self.inputs)
            self.n = len(self.inputs)
            with self.redis_server.pipeline() as pipe: # n inputs, n models
                for i in range(len(self.model_weights)):
                    self.model.set_weights(self.model_weights[i])
                    pipe.hset('model', i, pickle.dumps(self.model))
                for i in range(len(self.inputs)):
                    pipe.hset('input', i, pickle.dumps(self.inputs[i]))
                pipe.execute()

    def compute(self, layer_idx, epsilon=1e-7):
        P = []
        # run subprocess to get predictions
        if self.mut_level == 'i':
            for i in range(len(self.inputs)):
                cmd_1 = get_prediction_cmd(self.backend_1, self.db_flag, 0, i, i, layer_idx)
                cmd_2 = get_prediction_cmd(self.backend_2, self.db_flag, 0, i, i, layer_idx)
                p1 = Process(target=lambda: os.system(cmd_1))
                p2 = Process(target=lambda: os.system(cmd_2))
                p1.start()
                p2.start()
                P.append(p1)
                P.append(p2)

        elif self.mut_level == 'w':
            for i in range(len(self.model_weights)):
                cmd_1 = get_prediction_cmd(self.backend_1, self.db_flag, i, 0, i, layer_idx)
                cmd_2 = get_prediction_cmd(self.backend_2, self.db_flag, i, 0, i, layer_idx)
                p1 = Process(target=lambda: os.system(cmd_1))
                p2 = Process(target=lambda: os.system(cmd_2))
                p1.start()
                p2.start()
                P.append(p1)
                P.append(p2)

        elif self.mut_level == 'i+w':
            for i in range(len(self.model_weights)):
                cmd_1 = get_prediction_cmd(self.backend_1, self.db_flag, i, i, i, layer_idx)
                cmd_2 = get_prediction_cmd(self.backend_2, self.db_flag, i, i, i, layer_idx)
                p1 = Process(target=lambda: os.system(cmd_1))
                p2 = Process(target=lambda: os.system(cmd_2))
                p1.start()
                p2.start()
                P.append(p1)
                P.append(p2)

        for p in P:
            p.join()

        # load predictions
        with self.redis_server.pipeline() as pipe:
            for i in range(self.n):
                pipe.hget(f"predictions_{i}", self.backend_1)
                pipe.hget(f"predictions_{i}", self.backend_2)
            predictions = pipe.execute()
        
        predictions_1 = predictions[0::2]
        self.predictions_1 = []
        for p in predictions_1:
            p = pickle.loads(p)
            self.predictions_1.append(p / (np.linalg.norm(p) + epsilon))
        self.predictions_1 = np.concatenate(self.predictions_1)

        predictions_2 = predictions[1::2]
        self.predictions_2 = []
        for p in predictions_2:
            p = pickle.loads(p)
            self.predictions_2.append(p / (np.linalg.norm(p) + epsilon))
        self.predictions_2 = np.concatenate(self.predictions_2)

        assert len(self.predictions_1) == len(self.predictions_2)

        # compute fitness
        self.fitness_values = []
        for i in range(len(self.predictions_1)):
            d = np.abs(self.predictions_1[i] - self.predictions_2[i])
            self.fitness_values.append(np.sum(d) / len(d))

        return self.fitness_values

class NanFFunc:
    def __init__(self, redis_server, db_flag, mut_level, backend, model, model_weights, inputs):
        self.redis_server = redis_server
        self.db_flag = db_flag

        self.mut_level = mut_level
        self.backend = backend[0]
        self.model = model
        self.model_weights = model_weights
        self.inputs = inputs

    def prepare(self):
        if self.mut_level == 'i':
            self.n = len(self.inputs)
            with self.redis_server.pipeline() as pipe: # 1 model, n inputs
                pipe.hset('model', 0, pickle.dumps(self.model))
                for i in range(len(self.inputs)):
                    pipe.hset('input', i, pickle.dumps(self.inputs[i]))
                pipe.execute()
        elif self.mut_level == 'w':
            self.n = len(self.model_weights)
            with self.redis_server.pipeline() as pipe: # 1 input, n models
                pipe.hset('input', 0, pickle.dumps(self.inputs[0]))
                for i in range(len(self.model_weights)):
                    self.model.set_weights(self.model_weights[i])
                    pipe.hset('model', i, pickle.dumps(self.model))
                pipe.execute()
        elif self.mut_level == 'i+w':
            assert len(self.model_weights) == len(self.inputs)
            self.n = len(self.inputs)
            with self.redis_server.pipeline() as pipe: # n inputs, n models
                for i in range(len(self.model_weights)):
                    self.model.set_weights(self.model_weights[i])
                    pipe.hset('model', i, pickle.dumps(self.model))
                for i in range(len(self.inputs)):
                    pipe.hset('input', i, pickle.dumps(self.inputs[i]))
                pipe.execute()
    
    def compute(self, layer_idx, epsilon=1e-7):
        P = []
        # run subprocess to get predictions
        if self.mut_level == 'i':
            for i in range(len(self.inputs)):
                cmd_1 = get_prediction_cmd(self.backend, self.db_flag, 0, i, i, layer_idx)
                p1 = Process(target=lambda: os.system(cmd_1))
                p1.start()
                P.append(p1)

        elif self.mut_level == 'w':
            for i in range(len(self.model_weights)):
                cmd_1 = get_prediction_cmd(self.backend, self.db_flag, i, 0, i, layer_idx)
                p1 = Process(target=lambda: os.system(cmd_1))
                p1.start()
                P.append(p1)

        elif self.mut_level == 'i+w':
            for i in range(len(self.model_weights)):
                cmd_1 = get_prediction_cmd(self.backend, self.db_flag, i, i, i, layer_idx)
                p1 = Process(target=lambda: os.system(cmd_1))
                p1.start()
                P.append(p1)

        
        for p in P:
            p.join()

        # load predictions
        with self.redis_server.pipeline() as pipe:
            for i in range(self.n):
                pipe.hget(f"predictions_{i}", self.backend)
            predictions = pipe.execute()
        
        self.predictions = []
        for p in predictions:
            self.predictions.append(pickle.loads(p))
        self.predictions = np.concatenate(self.predictions)

        # compute fitness
        predictions_flatten = self.predictions.reshape((self.predictions.shape[0], -1))
        self.fitness_values = np.amax(predictions_flatten, axis=1) - np.amin(predictions_flatten, axis=1)
        
        return self.fitness_values