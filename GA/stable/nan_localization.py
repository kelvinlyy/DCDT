import redis
import keras
import pickle
import os
import functools
import numpy as np
from inc_localize import IncLocaliser
from generate_model_configs import model_configs
from kerasPredictCMD import get_outputs_cmd, get_prediction_cmd

'''
class for localising NaN error found in GA

parameters:
- ga: GA object
- db_flag: redis database number
'''
class NaN_Localizer:
    def __init__(self, ga, model_name, db_flag, save_path='localized_inc'):
        self.F = ga.F
        self.ga = ga
        self.input = ga.input
        self.model_name = model_name
        
        # deep copy model
        self.model = keras.models.clone_model(ga.model)
        model_weights = ga.model.get_weights()
        self.model.set_weights(model_weights)
        
        self.backend_1, self.backend_2 = ga.backends
        self.mut_level = ga.mut_level
        
        self.db_flag = db_flag
        self.redis_server = redis.Redis(db=self.db_flag)
        if not self.redis_server.ping():
            raise Exception("Redis server not set up")
            
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            
        self.save_path = '/'.join([save_path, model_name])
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        
        # localisation results
        self.L = [] # localized layers with backend
        
    # compute population output at layer l
    def get_p_output(self, p, backend, l):
        formatted_P = self.ga.formatPopulations([p])[0] # convert population into inputs, model weights or both
        if self.mut_level == 'i':
            self.input = formatted_P[0]
        elif self.mut_level == 'w':
            self.model.set_weights(formatted_P[1])
        elif self.mut_level == 'i+w':
            self.input = formatted_P[0]
            self.model.set_weights(formatted_P[1])
            
        with self.redis_server.pipeline() as pipe:
            pipe.hset('model', 0, pickle.dumps(self.model))
            pipe.hset('input', 0, pickle.dumps(self.input))
            pipe.execute()

        # run program to compute outputs
        cmd = get_outputs_cmd(backend, self.db_flag, 0, 0, 0, l)
        os.system(cmd)

        # get output
        output = pickle.loads(self.redis_server.hget("predictions_0", backend))

        return output
    
    # get the exclusive nan layers of the 2 backends
    def get_inconsistent_nan_layers(self, f):
        if len(f[0]) == 1: # only one framework triggers nan
            if f[0][1] == self.backend_1:
                return f[0][2], []
            else:
                return [], f[0][2]
            
        e1, e2 = f[0]
        inc_nan_l1 = list(set(e1[2]).difference(set(e2[2]))) # exclusive nan layers from backend 1
        inc_nan_l2 = list(set(e2[2]).difference(set(e1[2]))) # exclusive nan layers from backend 2
        return inc_nan_l1, inc_nan_l2
          
    # compute output from a layer
    def get_layer_output(self, backend, l, o1):
        m = keras.Sequential()
        m.add(self.model.layers[l]) # one-layer model
        self.redis_server.hset('model', 0, pickle.dumps(m))
        self.redis_server.hset('input', 0, pickle.dumps(o1))
        
        # get output
        cmd = get_prediction_cmd(backend, self.db_flag, 0, 0, 0)
        os.system(cmd)
        
        output = pickle.loads(self.redis_server.hget("predictions_0", backend))
        
        return output
    
    # merge and sort exclusive nan in the 2 backends     
    def merge_exclusive_nan(self, a, b):
        i = j = 0
        c = []
        while i < len(a) and j < len(b):
            if a[i] < b[j]:
                c.append(a[i])
                i += 1
            else:
                c.append(b[j])
                j += 1

        if i < len(a):
            c.extend(a[i:])

        if j < len(b):
            c.extend(b[j:])

        return c

    # localize nan inconsistencies by substitution
    def nan_sub_localize(self, f):
        L_nan = [] # localized nan inconsistency
        inc_nan_l1, inc_nan_l2 = self.get_inconsistent_nan_layers(f)

        print('Localizing NaN inconsistencies by substitution...')
        if inc_nan_l1 != []:
            for l in inc_nan_l1:
                o1 = self.get_p_output(f[1], self.backend_2, l-1) # output at layer l-1 from backend_2
                o2 = self.get_layer_output(self.backend_1, l, o1) # output at layer l from backend_1 using o1 as input
                if np.isnan(o2).any(): # nan persists after the change
                    print(f'NaN inconsistencies found in layer {l}')
                    L_nan.append([self.backend_1, l])
                
        if inc_nan_l2 != []:
            for l in inc_nan_l2:
                o1 = self.get_p_output(f[1], self.backend_1, l-1) # output at layer l-1 from backend_1
                o2 = self.get_layer_output(self.backend_2, l, o1) # output at layer l from backend_2 using o1 as input
                if np.isnan(o2).any(): # nan persists after the change
                    print(f'NaN inconsistencies found in layer {l}')
                    L_nan.append([self.backend_2, l])
                    
        return L_nan, inc_nan_l1, inc_nan_l2

    # localize inconsistencies by calculating layer distance
    def inc_check_localize(self, f, inc_nan_l1, inc_nan_l2, t1, t2):
        exclusive_nan = self.merge_exclusive_nan(inc_nan_l1, inc_nan_l2)
        L_inc = []
        
        start_layer = 0
        for l in exclusive_nan:
            test_model = keras.Sequential()
            for i in range(start_layer, l):
                test_model.add(self.model.layers[i])
                
            test_model.build(model.layers[start_layer].get_input_at(0).shape)
            model_config = model_configs(test_model)
            start_layer = l
            
            self.incL = IncLocaliser(test_model, [self.backend_1, self.backend_2], self.input, model_config, self.db_flag+1)
            
            X, Y, M = self.incL.main(t1, t2)
            if X == [] and Y == [] and M == []:
                continue
            else:
                L_inc.append([X, Y, M])
        
        return L_inc
        
        
        
    def localize(self, t1, t2):
        for i, f in enumerate(self.F):
            L_nan, inc_nan_l1, inc_nan_l2 = self.nan_sub_localize(f)
                
            if L_nan != []: # localized nan inconsistency
                self.L.append([i, 'nan_sub', L_nan])
                continue
            elif inc_nan_l1 == [] and inc_nan_l2 == []:
                continue
            else:
                L_inc = self.inc_check_localize(f, inc_nan_l1, inc_nan_l2, t1, t2)
                if L_inc != []:
                    self.L.append([i, 'inc_check', L_inc])
                
            
           
        return self.L
    