import redis
import keras
import pickle
import os
import numpy as np
from kerasPredictCMD import get_outputs_cmd, get_prediction_cmd

'''
class for localising NaN error found in GA

parameters:
- ga: GA object
- db_flag: redis database number
'''
class SourceLocaliser_F:
    def __init__(self, ga, db_flag):
        self.F = ga.F
        self.ga = ga
        
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
        
        # localisation results
        self.L = [] # localized layers with backend
        self.UF = [] # unsolved failed cases, need manual investigation
        
    # compute population output at layer l
    def get_p_output(self, p, backend, l):
        formatted_P = self.ga.formatPopulations([p])[0] # convert population into inputs, model weights or both
        if self.mut_level == 'i':
            with self.redis_server.pipeline() as pipe:
                pipe.hset('model', 0, pickle.dumps(self.model))
                pipe.hset('input', 0, pickle.dumps(formatted_P[0]))
                pipe.execute()

        elif self.mut_level == 'w':
            self.model.set_weights(formatted_P[1])
            with self.redis_server.pipeline() as pipe:
                pipe.hset('model', 0, pickle.dumps(self.model))
                pipe.hset('input', 0, pickle.dumps(self.input))
                pipe.execute()

        elif self.mut_level == 'i+w':
            self.model.set_weights(formatted_P[1])
            with self.redis_server.pipeline() as pipe:
                pipe.hset('model', 0, pickle.dumps(self.model))
                pipe.hset('input', 0, pickle.dumps(formatted_P[0]))
                pipe.execute()

        # run program to compute outputs
        cmd = get_outputs_cmd(backend, self.db_flag, 0, 0, 0, l)
        os.system(cmd)

        # get output
        output = pickle.loads(self.redis_server.hget("predictions_0", backend))

        return output
    
    # get the exclusive nan layers of the 2 backends
    def get_inconsistent_nan_layers(self, f):
        e1, e2 = f[0]
        inc_nan_l1 = list(set(e1[2]).difference(set(e2[2]))) # exclusive nan layers from backend 1
        inc_nan_l2 = list(set(e2[2]).difference(set(e1[2]))) # exclusive nan layers from backend 2
        return inc_nan_l1, inc_nan_l2
    
    # main function to run the localisation algorithm
    def localiseNan(self):
        for f in self.F:
            if len(f[0]) == 1: # only 1 framework triggers the error
                if f[0][0][0] == 'nan':
                    self.nan_localize_alg_1(f)
                else:
                    self.UF.append(['NN', f]) # non nan error
                    
            elif len(f[0]) == 2: # both frameworks trigger errors
                err_1, err_2 = f[0]
                if err_1[0] == 'nan' and err_2[0] == 'nan':
                    if err_1[2] != err_2[2]: # there is at least one exclusive nan layer in the frameworks
                        self.nan_localize_alg_2(f)
                else:
                    self.UF.append(['NN', f]) # non nan errors
         
        return self.L, self.UF
                
    # compute output from a layer
    def get_layer_output(self, backend, l, o1):
        m = keras.Sequential()
        m.add(self.model.layers[l]) # one-layer model
        self.redis_server.hset('model', 0, pickle.dumps(m))
        self.redis_server.hset('input', 0, pickle.dumps(o1))
        
        # get output
        cmd = get_prediction_cmd(backend, self.db_flag, 0, 0, 0, 0)
        os.system(cmd)
        
        output = pickle.loads(self.redis_server.hget("predictions_0", backend))
        
        return output
    
    # localize nan for 1 backend
    def nan_localize_alg_1(self, f):
        _, backend_1, nan_l = f[0][0]
        backend_2 = self.backend_1 if backend_1 == self.backend_2 else self.backend_2 
        
        for l in nan_l:
            o1 = self.get_p_output(f[1], backend_2, l-1) # output at layer l-1 from backend_2
            o2 = self.get_layer_output(backend_1, l, o1) # output at layer l from backend_1 using o1 as input
            if np.isnan(o2).any(): # nan persists after the change
                self.L.append([backend_1, l])
            else:
                self.UF.append(['IncP', backend_1, l, f[1]]) # inconsistency in previous layers
                
    # localize nan for 2 backends
    def nan_localize_alg_2(self, f):
        inc_nan_l1, inc_nan_l2 = self.get_inconsistent_nan_layers(f)

        if inc_nan_l1 != []:
            for l in inc_nan_l1:
                o1 = self.get_p_output(f[1], self.backend_2, l-1) # output at layer l-1 from backend_2
                o2 = self.get_layer_output(self.backend_1, l, o1) # output at layer l from backend_1 using o1 as input
                if np.isnan(o2).any(): # nan persists after the change
                    self.L.append([self.backend_1, l])
                else:
                    self.UF.append(['IncP', self.backend_1, l, f[1]]) # inconsistency in previous layers
                
        if inc_nan_l2 != []:
            for l in inc_nan_l2:
                o1 = self.get_p_output(f[1], self.backend_1, l-1) # output at layer l-1 from backend_1
                o2 = self.get_layer_output(self.backend_2, l, o1) # output at layer l from backend_2 using o1 as input
                if np.isnan(o2).any(): # nan persists after the change
                    self.L.append([self.backend_2, l])
                else:
                    self.UF.append(['IncP', self.backend_2, l, f[1]]) # inconsistency in previous layers
    