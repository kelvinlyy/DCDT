import os
import redis
import keras
import pickle
import functools
import numpy as np

from Util.kerasPredictCMD import get_outputs_cmd, get_prediction_cmd

'''
class for localising NaN error found in GA

parameters:
- ga: GA object
- db_flag: redis database number
'''
class NanLocalizer:
    def __init__(self, ga, model_name, db_flag, savedFailed_dir='localized_nan'):
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
            
        if not os.path.exists(savedFailed_dir):
            os.mkdir(savedFailed_dir)
            
        self.savedFailed_dir = '/'.join([savedFailed_dir, model_name])
        if not os.path.exists(self.savedFailed_dir):
            os.mkdir(self.savedFailed_dir)
        
        # localisation results
        self.L = [] # localized layers with backend
        self.UF = [] # unsolved failed cases, need manual investigation
        
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
    
    # localize nan for 1 backend
    def nan_localize_alg_1(self, i, f):
        L = []
        UF = []
        _, backend_1, nan_l = f[0][0]
        backend_2 = self.backend_1 if backend_1 == self.backend_2 else self.backend_2 
        
        for l in nan_l:
            o1 = self.get_p_output(f[1], backend_2, l-1) # output at layer l-1 from backend_2
            o2 = self.get_layer_output(backend_1, l, o1) # output at layer l from backend_1 using o1 as input
            if np.isnan(o2).any(): # nan persists after the change
                L.append([i, backend_1, l])
            else:
                UF.append([i, backend_1, l])
                
        return L, UF
                
    # localize nan for 2 backends
    def nan_localize_alg_2(self, i, f):
        L = []
        UF = []
        inc_nan_l1, inc_nan_l2 = self.get_inconsistent_nan_layers(f)

        if inc_nan_l1 != []:
            for l in inc_nan_l1:
                o1 = self.get_p_output(f[1], self.backend_2, l-1) # output at layer l-1 from backend_2
                o2 = self.get_layer_output(self.backend_1, l, o1) # output at layer l from backend_1 using o1 as input
                if np.isnan(o2).any(): # nan persists after the change
                    L.append([i, self.backend_1, l])
                else:
                    UF.append([i, self.backend_1, l])
                
        if inc_nan_l2 != []:
            for l in inc_nan_l2:
                o1 = self.get_p_output(f[1], self.backend_1, l-1) # output at layer l-1 from backend_1
                o2 = self.get_layer_output(self.backend_2, l, o1) # output at layer l from backend_2 using o1 as input
                if np.isnan(o2).any(): # nan persists after the change
                    L.append([self.backend_2, l])
                else:
                    UF.append([i, self.backend_2, l])
                    
        return L, UF
    
    # main function to run the localisation algorithm
    def localiseNan(self, saveDisk=True):
        failed_dirs = [int(f) for f in os.listdir(self.savedFailed_dir) \
                       if os.path.isdir(os.path.join(self.savedFailed_dir, f)) and f[0].isnumeric()]
        if failed_dirs == []:
            dir_i = 0
        else:
            dir_i = sorted(failed_dirs)[-1] + 1 # new save directory name if saveDisk=True
            
        for i in range(len(self.F)):
            f = self.F[i]
            L = []
            UF = []
            if len(f[0]) == 1: # only 1 framework triggers the error
                if f[0][0][0] == 'nan':
                    L, UF = self.nan_localize_alg_1(i, f)
                    self.L.extend(L)
                    self.UF.extend(UF)
                else:
                    UF = [[i, 'non-nan']] # non nan error
                    self.UF.extend(UF)
                    
            elif len(f[0]) == 2: # both frameworks trigger errors
                err_1, err_2 = f[0]
                if err_1[0] == 'nan' and err_2[0] == 'nan':
                    if err_1[2] != err_2[2]: # there is at least one exclusive nan layer in the frameworks
                        L, UF = self.nan_localize_alg_2(i, f)
                        self.L.extend(L)
                        self.UF.extend(UF)
                else:
                    UF = [[i, 'non-nan']] # non nan errors
                    self.UF.extend(UF)
                    
            if saveDisk: # save failed cases to disk
                if L == []:
                    continue
                    
                dir_path = os.path.join(self.savedFailed_dir, str(dir_i))
                os.mkdir(dir_path)
                # write failed info into the files
                info_f, weight_f, input_f = map(functools.partial(os.path.join, dir_path), ['info.txt', 'weight', 'input'])
                with open(info_f, 'w') as f:
                    f.write('NaN\n')
                    f.write('Localized layers: {}\n'.format(L))
                    f.write('Unsolved: {}'.format(UF))
                    
                self.model.save_weights(weight_f)
                np.save(input_f, self.input)

                dir_i += 1

            
           
        return self.L, self.UF
    