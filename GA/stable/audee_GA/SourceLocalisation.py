import redis
import pickle
from kerasPredictCMD import get_outputs_cmd, get_prediction_cmd
from InconsistencyCheck import ga_inc
from multiprocessing import Process
import keras
import os
import sys
from tqdm import tqdm
import numpy as np
import random
import time

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        
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
        self.redis_server.hset("model", 0, pickle.dumps(model))
        
    def update_x(self, x):
        self.redis_server.hset("input", 0, pickle.dumps(x))
        
        
    def compute_layer_dist(self, f_prime, x_prime):
        # load f_prime, x_prime into redis db
        self.update_x(x_prime)
        self.update_model(f_prime)

        cmd_1 = get_prediction_cmd(self.backend_1, self.db_flag, 0, 0, 0)
        cmd_2 = get_prediction_cmd(self.backend_2, self.db_flag, 0, 0, 0)

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

        p1 = pickle.loads(predictions[0])
        p2 = pickle.loads(predictions[1])

        assert len(p1) == len(p2)

        predictions_diff = np.abs(p1 - p2).ravel()
        layer_dist = np.sum(predictions_diff) / len(predictions_diff)
        
        # recover model and input state
        self.update_x(self.x)
        self.update_model(self.model)

        return layer_dist
            
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
#             p1 = (p1 - np.min(p1)) / (np.max(p1) - np.min(p1))
#             p2 = (p2 - np.min(p2)) / (np.max(p2) - np.min(p2))
                  
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
    def replace(self, L, L_idx, a):
        layer_config = L.get_config()
        original_a = layer_config[a]
        alt_a = [v for v in self.model_config[L_idx][a] if v != original_a]
        if not alt_a:
            return None, None
        new_a = random.choice(alt_a)
        layer_config[a] = new_a
        L_prime = L.from_config(layer_config)
        return L_prime, new_a
    
    # create a simple model using 1 layer L_prime 
    def create_test_model(self, L_prime, x_prime):
        test_model = keras.Sequential()
        test_model.add(L_prime)
        test_model.build(x_prime.shape)
        return test_model
    
    # check errors while running the newly created model
    def checkCrash_NaN(self, f_prime, x_prime):
        self.update_model(f_prime)
        self.update_x(x_prime)
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
    
    def fixDNN(self, L, L_idx, a, a_prime):
        layer_config = L.get_config()
        layer_config[a] = a_prime
        L_prime = L.from_config(layer_config)
        x = L_prime(self.model.layers[L_idx-1].get_output_at(-1))

        for i in range(L_idx+1, len(self.model.layers)):
            x = self.model.layers[i](x)

        f = keras.models.Model(input=self.model.input, output=x)
        f.build(self.x.shape)
        
        return f
    
    def detectInconsistency(self, f_prime, x_prime):
        with HiddenPrints():
            f_prime.predict(x_prime) # init model output layers
            x_max = ga_inc(self.backend_1, self.backend_2, f_prime, x_prime, 1, self.db_flag+1)

        return x_max
    
    def main(self, t1, t2, epsilon=10**-7):
        X = []
        Y = []
        
        visited_layers = []
        
        for _ in range(8): # max # layers to localize
            start_time = time.perf_counter()
            beta = [l for l in self.t1_dists_change(t1) if l not in visited_layers]
            if beta == []: # finish localization
                return X, Y
            
            L, L_idx = beta[0] # get layer object and layer index
            a_L = self.model_config[L_idx] # get the set of possible layer parameters
            visited_layers.append(beta[0])
            
            print(f'Localizing layer {L_idx}: {L.__class__.__name__}...')
            
            if not a_L:
                print('Localization failed: No parameter received\n')
                print()
                continue
            # update inputs to fit the chosen layer
            x_prime = pickle.loads(self.redis_server.hget("predictions_0", self.backend_1))[L_idx-1]
            self.update_x(x_prime)
            
            P = []
            for a in a_L:
                L_prime, a_prime = self.replace(L, L_idx, a)
                if a_prime == None: # no available parameter
                    continue
                    
                f_prime = self.create_test_model(L_prime, x_prime)
                
                print(f'Parameter "{a}" is set to be "{a_prime}"')

                y = self.checkCrash_NaN(f_prime, x_prime)
                if y != []:
                    print('Errors: model crashes or gives NaN')
                    Y.append([y, L_idx, a_prime])
                else:
                    print('Maximizing inconsistency of the input...')
                    
                    x_max = self.detectInconsistency(f_prime, x_prime)
                    
                    L_dist = self.compute_layer_dist(f_prime, x_max)
                    max_prev_layers_dist = max(self.layers_dist[:L_idx])
                    L_dist_change = (L_dist - max_prev_layers_dist) / (max_prev_layers_dist + epsilon)

                    if L_dist_change < t2:
                        print(f'Inconsistency is localized in: \t{a} = {L.get_config()[a]}')
                        P.append(a)
                        
                        self.model = self.fixDNN(L, L_idx, a, a_prime)
                        self.update_model(self.model)
             
            if P != []:
                X.append([L_idx, P])
            
            end_time = time.perf_counter()
            print()
            print(f'Time taken: {end_time - start_time}')
            print()
            print()
                
        return X, Y