import redis
import pickle
from kerasPredictCMD import get_outputs_cmd
from multiprocessing import Process
import keras
import os
from tqdm import tqdm
import numpy as np

class SourceLocaliser:
    def __init__(self, model, frameworks, x, model_config, db_flag):
        self.redis_server = redis.Redis(db=db_flag)
        self.model = model
        self.backend_1, self.backend_2 = frameworks
        self.x = x
        self.model_config = model_config
        self.db_flag = db_flag
        
    def prepare(self):
        with self.redis_server.pipeline() as pipe:
            pipe.hset("model", 0, pickle.dumps(self.model))
            pipe.hset("input", 0, pickle.dumps(self.x))
            pipe.execute()
            
    def update_model(self, model):
        self.redis_server.mset({"model": pickle.dumps(model)})
        
    def update_x(self, x):
        self.redis_server.mset({"inputs": pickle.dumps(x)})
            
    # compute inconsistency fitness score
    def compute_all_layers_dist(self):
        cmd_1 = get_outputs_cmd(self.backend_1, self.db_flag, 0, 0, 0, 'all')
        cmd_2 = get_outputs_cmd(self.backend_2, self.db_flag, 0, 0, 0, 'all')

        p1 = Process(target=lambda: os.system(cmd_1))
        p2 = Process(target=lambda: os.system(cmd_2))
        p1.start()
        p2.start()
        p1.join()
        p2.join()
        
        # load predictions
        with self.redis_server.pipeline() as pipe:
            pipe.hget("predictions_0", self.backend_1)
            pipe.hget("predictions_0", self.backend_2)
            predictions = pipe.execute()

        predictions_1 = pickle.loads(predictions[0])
        predictions_2 = pickle.loads(predictions[1])
        
        assert len(predictions_1) == len(predictions_2)
        
        self.layers_dist = []
        for i in range(len(predictions_1)):
            predictions_diff = np.abs(predictions_1[i] - predictions_2[i]).ravel()
            self.layers_dist.append(np.sum(predictions_diff) / len(predictions_diff))
        
        return self.layers_dist
    
    # compute rate of change of inconsistency fitness score betwen layers
    def compute_dists_change(self, layer_idx, epsilon=10**-7):
        max_prev_layers_dist = max(self.layers_dist[:layer_idx])
        self.dists_change = (self.layers_dist[layer_idx] - max_prev_layers_dist) / (max_prev_layers_dist + epsilon)
        return self.dists_change
    
    
    # return only layer indexes with rate of change larger than t1
    def t1_dists_change(self, t1):
        self.compute_all_layers_dist() # prepare for the subsequent computations
        self.t1_layers = []
        for i in range(2, len(self.model.layers)): # choose 2 to escape the distance jump from the input layer
            layer_change = self.compute_dists_change(i)
            if layer_change >= t1:
                self.t1_layers.append([self.model.layers[i], i])
        return self.t1_layers
    
    # create a new layer based on config
    def replace(self, L, new_config):
        L_prime = L.from_config(new_config)
        return L_prime
    
    # create a simple model using 1 layer L_prime 
    def create_test_model(self, L_prime):
        test_model = keras.Sequential()
        test_model.add(L_prime)
        # not sure if need to call model.build()
        return test_model
    
    # check errors while running the newly created model
    def checkCrash_NaN(self, f_prime):
        self.update_model(f_prime)
        cmd_1 = get_outputs_cmd(self.backend_1, self.db_flag, 0, 0, 0, 'errors_0')
        cmd_2 = get_outputs_cmd(self.backend_1, self.db_flag, 0, 0, 0, 'errors_0')

        p1 = Process(target=lambda: os.system(cmd_1))
        p2 = Process(target=lambda: os.system(cmd_2))
        p1.start()
        p2.start()
        p1.join()
        p2.join()
        
        # load errors
        with self.redis_server.pipeline() as pipe:
            pipe.hget("errors_0", self.backend_1)
            pipe.hget("errors_0", self.backend_2)
            errors = pipe.execute()
            
        errors_1 = pickle.loads(errors[0])
        errors_2 = pickle.loads(errors[1])
        
        errors = []
        if errors_1 != []:
            errors.append([errors_1, self.backend_1])
            
        if errors_2 != []:
            errors.append([errors_1, self.backend_2])
            
        return errors
    
    def fixDNN(self, f, X, L_prime, L_idx):
        pass
    
    def main(self, t1, t2):
        X = []
        Y = []
        
        for _ in tqdm(range(5)):
            beta = self.t1_dists_change(t1)
            if beta == []: # finish localization
                return X, Y
            
            L, L_idx = beta[0]
            
            if L_idx in self.model_config:
                a_L = self.model_config[L_idx]
            else:
                continue
            
            P = []
            for a in a_L:
                L_prime = self.replace(L, a)
                f_prime = self.create_test_model(L_prime)
                
                # update inputs to fit the newly created model f_prime
                x_prime = self.redis_server.hget("predictions", self.backend_1)[L_idx-1]
                self.update_x(x_prime)
                
                y = self.checkCrash_NaN(f_prime)
                if y != []:
                    Y.append([y, L_idx, a])
                else:
                    continue
                    x_max = detectInconsistency(f_prime) # when to stop?
                    self.update_x(x_max)
                    
                    if self.compute_dists_change(L_idx) < t2:
                        P.append(a)
                    
                    if P != []:
                        X.append([L, P]) # why keep appending P
                        
                    f = self.fixDNN(f, X, L_prime, L_idx)
                
        return X, Y