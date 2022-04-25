import os
import pickle
import numpy as np
from multiprocessing import Process

from Util.kerasPredictCMD import get_outputs_cmd, get_coverage_cmd


P_NUM = 8 # number of processes running simultaneously

class CoverageFFunc:
    def __init__(self, redis_server, db_flag, mut_level, backend, model, model_weights, inputs):
        self.redis_server = redis_server
        self.db_flag = db_flag

        self.mut_level = mut_level
        self.backend = backend[0]
        self.model = model
        self.model_weights = model_weights
        self.inputs = inputs

    # send the models, inputs to be used to the redis database
    def prepare(self):
        if self.mut_level == 'i': # input-level mutation
            self.n = len(self.inputs)
            with self.redis_server.pipeline() as pipe: # 1 model, n inputs
                pipe.hset('model', 0, pickle.dumps(self.model))
                for i in range(len(self.inputs)):
                    pipe.hset('input', i, pickle.dumps(self.inputs[i]))
                pipe.execute()
                
        elif self.mut_level == 'w': # weight-level mutation
            self.n = len(self.model_weights)
            with self.redis_server.pipeline() as pipe: # 1 input, n models
                pipe.hset('input', 0, pickle.dumps(self.inputs[0]))
                for i in range(len(self.model_weights)):
                    self.model.set_weights(self.model_weights[i])
                    pipe.hset('model', i, pickle.dumps(self.model))
                pipe.execute()
                
        elif self.mut_level == 'i+w': # input and weight-level mutation
            assert len(self.model_weights) == len(self.inputs)
            self.n = len(self.inputs)
            with self.redis_server.pipeline() as pipe: # n inputs, n models
                for i in range(len(self.model_weights)):
                    self.model.set_weights(self.model_weights[i])
                    pipe.hset('model', i, pickle.dumps(self.model))
                for i in range(len(self.inputs)):
                    pipe.hset('input', i, pickle.dumps(self.inputs[i]))
                pipe.execute()

    # get back predictions from redis database and compute the inconsistency fitness values
    def compute(self, layer_idx, epsilon=1e-7):
        P = []
        # run multi-processes to get predictions
        if self.mut_level == 'i': # input-level mutation
            for i in range(len(self.inputs)):
                cmd_1 = get_coverage_cmd(self.backend, self.db_flag, 0, i, i)
                os.system(cmd_1)

        elif self.mut_level == 'w': # weight-level mutation
            for i in range(len(self.model_weights)):
                cmd_1 = get_coverage_cmd(self.backend, self.db_flag, i, 0, i)
                os.system(cmd_1)

        elif self.mut_level == 'i+w': # input and weight-level mutation
            for i in range(len(self.model_weights)):
                cmd_1 = get_coverage_cmd(self.backend, self.db_flag, i, i, i)
                os.system(cmd_1)

        # load predictions
        with self.redis_server.pipeline() as pipe:
            for i in range(self.n):
                pipe.hget(f"coverage_{i}", self.backend)
            coverages = pipe.execute()
            
        self.fitness_values = []
        for c in coverages:
            c = pickle.loads(c)
            self.fitness_values.append(c)
            
        return self.fitness_values
    
class InconsistencyFFunc:
    def __init__(self, redis_server, db_flag, mut_level, backends, model, model_weights, inputs):
        self.redis_server = redis_server
        self.db_flag = db_flag
        self.mut_level = mut_level
        self.backend_1, self.backend_2 = backends
        self.model = model
        self.model_weights = model_weights
        self.inputs = inputs

    # send the models, inputs to be used to the redis database
    def prepare(self):
        if self.mut_level == 'i': # input-level mutation
            self.n = len(self.inputs)
            with self.redis_server.pipeline() as pipe: # 1 model, n inputs
                pipe.hset('model', 0, pickle.dumps(self.model))
                for i in range(len(self.inputs)):
                    pipe.hset('input', i, pickle.dumps(self.inputs[i]))
                pipe.execute()
                
        elif self.mut_level == 'w': # weight-level mutation
            self.n = len(self.model_weights)
            with self.redis_server.pipeline() as pipe: # 1 input, n models
                pipe.hset('input', 0, pickle.dumps(self.inputs[0]))
                for i in range(len(self.model_weights)):
                    self.model.set_weights(self.model_weights[i])
                    pipe.hset('model', i, pickle.dumps(self.model))
                pipe.execute()
                
        elif self.mut_level == 'i+w': # input and weight-level mutation
            assert len(self.model_weights) == len(self.inputs)
            self.n = len(self.inputs)
            with self.redis_server.pipeline() as pipe: # n inputs, n models
                for i in range(len(self.model_weights)):
                    self.model.set_weights(self.model_weights[i])
                    pipe.hset('model', i, pickle.dumps(self.model))
                for i in range(len(self.inputs)):
                    pipe.hset('input', i, pickle.dumps(self.inputs[i]))
                pipe.execute()

    # get back predictions from redis database and compute the inconsistency fitness values
    def compute(self, layer_idx, normalize=False, epsilon=1e-7):
        P = []
        # run multi-processes to get predictions
        if self.mut_level == 'i': # input-level mutation
            for i in range(len(self.inputs)):
                cmd_1 = get_outputs_cmd(self.backend_1, self.db_flag, 0, i, i, layer_idx)
                cmd_2 = get_outputs_cmd(self.backend_2, self.db_flag, 0, i, i, layer_idx)
                p1 = Process(target=lambda: os.system(cmd_1))
                p2 = Process(target=lambda: os.system(cmd_2))
                p1.start()
                p2.start()
                P.append(p1)
                P.append(p2)
                if (i+1)%(P_NUM//2) == 0:
                    for p in P:
                        p.join()
                        
                    P = []



        elif self.mut_level == 'w': # weight-level mutation
            for i in range(len(self.model_weights)):
                cmd_1 = get_outputs_cmd(self.backend_1, self.db_flag, i, 0, i, layer_idx)
                cmd_2 = get_outputs_cmd(self.backend_2, self.db_flag, i, 0, i, layer_idx)
                p1 = Process(target=lambda: os.system(cmd_1))
                p2 = Process(target=lambda: os.system(cmd_2))
                p1.start()
                p2.start()
                P.append(p1)
                P.append(p2)
                if (i+1)%(P_NUM//2) == 0:
                    for p in P:
                        p.join()
                        
                    P = []

        elif self.mut_level == 'i+w': # input and weight-level mutation
            for i in range(len(self.model_weights)):
                cmd_1 = get_outputs_cmd(self.backend_1, self.db_flag, i, i, i, layer_idx)
                cmd_2 = get_outputs_cmd(self.backend_2, self.db_flag, i, i, i, layer_idx)
                p1 = Process(target=lambda: os.system(cmd_1))
                p2 = Process(target=lambda: os.system(cmd_2))
                p1.start()
                p2.start()
                P.append(p1)
                P.append(p2)
                if (i+1)%(P_NUM//2) == 0:
                    for p in P:
                        p.join()
                        
                    P = []

        for p in P: # wait for the processes to be executed
            p.join()

        # load predictions
        with self.redis_server.pipeline() as pipe:
            for i in range(self.n):
                pipe.hget(f"predictions_{i}", self.backend_1)
                pipe.hget(f"predictions_{i}", self.backend_2)
            predictions = pipe.execute()
        
        predictions_1 = predictions[0::2] # predictions by backend_1
        self.predictions_1 = []
        for p in predictions_1:
            p = pickle.loads(p)
            if normalize:
                p = (p - np.min(p)) / (np.max(p) - np.min(p) + epsilon)
                
            self.predictions_1.append(p)
        self.predictions_1 = np.squeeze(np.stack(self.predictions_1))

        predictions_2 = predictions[1::2] # predictions by backend_2
        self.predictions_2 = []
        for p in predictions_2:
            p = pickle.loads(p)
            if normalize:
                p = (p - np.min(p)) / (np.max(p) - np.min(p) + epsilon)
                
            self.predictions_2.append(p)
        self.predictions_2 = np.squeeze(np.stack(self.predictions_2))

        assert len(self.predictions_1) == len(self.predictions_2)

        # compute fitness
        self.fitness_values = []
        for i in range(len(self.predictions_1)):
            if np.all(np.isfinite(self.predictions_1)) and np.all(np.isfinite(self.predictions_1)): # if there is nan in the predictions
                d = np.abs(self.predictions_1[i] - self.predictions_2[i])
                self.fitness_values.append(np.sum(d) / len(d))
            else:
                self.fitness_values.append(np.nan)

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

    # send the models, inputs to be used to the redis database
    def prepare(self):
        if self.mut_level == 'i': # input-level mutation
            self.n = len(self.inputs)
            with self.redis_server.pipeline() as pipe: # 1 model, n inputs
                pipe.hset('model', 0, pickle.dumps(self.model))
                for i in range(len(self.inputs)):
                    pipe.hset('input', i, pickle.dumps(self.inputs[i]))
                pipe.execute()
                
        elif self.mut_level == 'w': # weight-level mutation
            self.n = len(self.model_weights)
            with self.redis_server.pipeline() as pipe: # 1 input, n models
                pipe.hset('input', 0, pickle.dumps(self.inputs[0]))
                for i in range(len(self.model_weights)):
                    self.model.set_weights(self.model_weights[i])
                    pipe.hset('model', i, pickle.dumps(self.model))
                pipe.execute()
                
        elif self.mut_level == 'i+w': # input and weight-level mutation
            assert len(self.model_weights) == len(self.inputs)
            self.n = len(self.inputs)
            with self.redis_server.pipeline() as pipe: # n inputs, n models
                for i in range(len(self.model_weights)):
                    self.model.set_weights(self.model_weights[i])
                    pipe.hset('model', i, pickle.dumps(self.model))
                for i in range(len(self.inputs)):
                    pipe.hset('input', i, pickle.dumps(self.inputs[i]))
                pipe.execute()
    
    # get back predictions from redis database and compute the nan fitness values
    def compute(self, layer_idx, epsilon=1e-7):
        P = []
        # run multi-processes to get predictions
        if self.mut_level == 'i': # input-level mutation
            for i in range(len(self.inputs)):
                cmd_1 = get_outputs_cmd(self.backend, self.db_flag, 0, i, i, layer_idx)
                p1 = Process(target=lambda: os.system(cmd_1))
                p1.start()
                P.append(p1)

        elif self.mut_level == 'w': # weight-level mutation
            for i in range(len(self.model_weights)):
                cmd_1 = get_outputs_cmd(self.backend, self.db_flag, i, 0, i, layer_idx)
                p1 = Process(target=lambda: os.system(cmd_1))
                p1.start()
                P.append(p1)

        elif self.mut_level == 'i+w': # input and weight-level mutation
            for i in range(len(self.model_weights)):
                cmd_1 = get_outputs_cmd(self.backend, self.db_flag, i, i, i, layer_idx)
                p1 = Process(target=lambda: os.system(cmd_1))
                p1.start()
                P.append(p1)

        for p in P: # wait processes to be executed
            p.join()

        # load predictions
        with self.redis_server.pipeline() as pipe:
            for i in range(self.n):
                pipe.hget(f"predictions_{i}", self.backend)
            predictions = pipe.execute()
        
        self.predictions = []
        for p in predictions:
            self.predictions.append(pickle.loads(p))
        self.predictions = np.squeeze(np.stack(self.predictions))

        # compute fitness
        predictions_flatten = self.predictions.reshape((self.predictions.shape[0], -1))
        self.fitness_values = np.amax(predictions_flatten, axis=1) - np.amin(predictions_flatten, axis=1)
        
        return self.fitness_values