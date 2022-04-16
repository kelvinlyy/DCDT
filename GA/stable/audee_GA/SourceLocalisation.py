import redis
import pickle
from kerasPredictCMD import get_outputs_cmd
from InconsistencyCheck import ga_inc
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
            
    # compute layer distance
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

        self.predictions_1 = pickle.loads(predictions[0])
        self.predictions_2 = pickle.loads(predictions[1])
        
        assert len(self.predictions_1) == len(self.predictions_2)
        
        self.layers_dist = []
        for i in range(len(self.predictions_1)):
            p1 = self.predictions_1[i]
            p2 = self.predictions_2[i]
            p1 = (p1 - np.min(p1)) / (np.max(p1) - np.min(p1))
            p2 = (p2 - np.min(p2)) / (np.max(p2) - np.min(p2))
                  
            predictions_diff = np.abs(p1 - p2).ravel()
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
    def replace(self, L, a):
        layer_config = L.get_config()
        original_a = layer_config[a]
        alt_a = [v for v in self.model_config[L_idx][a] if v != original_a]
        new_a = random.choice(alt_a)
        layer_config[a] = new_a
        L_prime = L.from_config(layer_config)
        return L_prime, new_a
    
    # create a simple model using 1 layer L_prime 
    def create_test_model(self, L_prime):
        test_model = keras.Sequential()
        test_model.add(L_prime)
        # not sure if need to call model.build()
        return test_model
    
    # check errors while running the newly created model
    def checkCrash_NaN(self, f_prime):
        self.update_model(f_prime)
        cmd_1 = get_outputs_cmd(self.backend_1, self.db_flag, 0, 0, 0, 'error')
        cmd_2 = get_outputs_cmd(self.backend_1, self.db_flag, 0, 0, 0, 'error')

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
    
    def fixDNN(self, L, a, a_prime):
        layer_config = L.get_config()
        layer_config[a] = a_prime
        L_prime = L.from_config(layer_config)
        
        f = self.create_test_model(L_prime)
        return f
       
    
    def detectInconsistency(f_prime, x_prime):
        model = f_prime
        x_max = ga_inc(self.backend_1, self.backend_2, f_prime, x_prime, 255, self.db_flag+1)
        return max_max
    
    def main(self, t1, t2):
        X = []
        Y = []
        
        for _ in tqdm(range(5)):
            beta = self.t1_dists_change(t1)
            if beta == []: # finish localization
                return X, Y
            
            L, L_idx = beta[0] # get layer object and layer index
            
            if L_idx < len(self.model_config):
                a_L = self.model_config[L_idx] # get the set of possible layer parameters
            else:
                continue
            
            P = []
            # update inputs to fit the chosen layer
            x_prime = self.redis_server.hget("predictions", self.backend_1)[L_idx-1]
            self.update_x(x_prime)
            for a in a_L:
                L_prime, a_prime = self.replace(L, a)
                f_prime = self.create_test_model(L_prime)

                y = self.checkCrash_NaN(f_prime)
                if y != []:
                    Y.append([y, L_idx, a_prime])
                else:
                    x_max = detectInconsistency(f_prime, x_prime)
                    self.update_x(x_max)
                    
                    if self.compute_dists_change(L_idx) < t2:
                        P.append(a)
                        X.append([L_idx, P])
                        f = self.fixDNN(L, a, a_prime)
                        self.update_model(f)
                    
            self.update_x(self.x)
                
        return X, Y