import keras
import redis
import pickle
import numpy as np
import random
import time
from FFunc_keras import InconsistencyFFunc, NanFFunc

class GA:
    '''
    arguments e.g.:
    fit: ["inc", ["tensorflow", "theano"]]
    model: model
    inputs: x_test
    init_input_mut: 0.1
    init_weight_mut: 0
    r1: 0.5
    r2: 0.1
    m: 5
    n: 1000
    db_flag: 0
    '''
    def __init__(self, fit, model, inputs, db_flag):
        self.fit_func, self.phi = fit # a tuple: (fitness function, list of DL framework(s) to be used)
        
        new_model = keras.models.clone_model(model)
        new_model.set_weights(model.get_weights()) # clone_model does not clone the weights
        self.model = new_model
        
        self.inputs = inputs
        self.input_shape = inputs.shape[-3:] # shape of an image

        self.db_flag = db_flag
        self.redis_server = redis.Redis(db=db_flag)
        if not self.redis_server.ping():
            raise Exception("Redis server not set up")
        
    # initialize inputs and model weights
    def initPopulation(self, init_input_mut, init_weight_mut, n):
        if init_input_mut == 0:
            assert self.inputs.ndim == 4 # make sure self.inputs is of multiple images
            
            if len(self.inputs) < 5:
                raise Exception("Number of images should be greater than m")
            self.mutated_inputs = self.inputs
        else:
            assert self.inputs.ndim == 3 # make sure self.inputs is of 1 image only
            
            self.mutated_inputs = []
            for _ in range(n):
                self.mutated_inputs.append(self.inputs)
            self.mutated_inputs = np.array(self.mutated_inputs)

            noise = np.random.standard_cauchy(self.mutated_inputs.shape)
            noise[0] = 0 # keep 1 original input

            self.mutated_inputs = (np.clip(self.mutated_inputs/255 + noise * init_input_mut, 0, 1) * 255).astype(int)
            
        # get model weights
        original_weights = np.array(self.model.get_weights(), dtype=object)
        self.mutated_weights = original_weights
        
        # weight level
        # set the model weights of the GA object to the mutated weights
        if init_weight_mut != 0 and init_weight_mut != None:
            self.mutated_weights = []
            for layer_weight in original_weights:
                self.mutated_weights.append(layer_weight + np.random.standard_cauchy(layer_weight.shape) * init_weight_mut)
            self.model.set_weights(self.mutated_weights)
                
        return [self.mutated_inputs, self.mutated_weights]

    # upload model and inputs to the designated redis server
    def prepareFitness(self):
        if self.fit_func == "inc":
            self.FFunc = InconsistencyFFunc(self.redis_server, self.db_flag, self.phi, self.model, self.mutated_inputs)
        elif self.fit_func == "nan":
            self.FFunc = NanFFunc(self.redis_server, self.db_flag, self.phi, self.model, self.mutated_inputs)
        self.FFunc.prepare()

    # compute fitness scores for each input
    def computeFitness(self, layer_idx):
        self.fitness_values = np.squeeze(self.FFunc.compute(layer_idx))
        return self.fitness_values
        
    # get the top k fitness values and the corresponding inputs
    def getTopK_Fit(self, k):
        topK_idx = np.argpartition(self.fitness_values, -k)[-k:]
        return [self.mutated_inputs[topK_idx], self.fitness_values[topK_idx]]
        
    # select m candidates for parents of next-generation inputs
    def select(self, m):
        selected_index = np.argpartition(self.fitness_values, -m)[-m:]
        self.selected_x = self.mutated_inputs[selected_index]
        return self.selected_x
    
    # select 2 parents from the candidates
    def selectParents(self):
        self.x1, self.x2 = random.sample(list(self.selected_x), 2)
        return self.x1, self.x2
    
    # return a flatten list of a crossover product of the selected parents
    def crossover(self, r1):
        x1_flatten = self.x1.flatten()
        x2_flatten = self.x2.flatten()

        x1_factor = np.random.choice(2, size=x1_flatten.shape, p=[1-r1, r1])
        x2_factor = 1 - x1_factor
        
        self.x_prime = x1_flatten * x1_factor + x2_flatten * x2_factor

        return self.x_prime
    
    # mutate the crossover product and reshape it as the shape of the input instance
    def mutate(self, mut_rate):
        self.x_2prime=(np.clip((self.x_prime/255 + np.random.standard_cauchy(self.x_prime.shape) * mut_rate),0,1) * 255).astype(int)
        self.x_2prime = self.x_2prime.reshape(self.input_shape)
        return self.x_2prime
    
    # update inputs on the redis server
    def update_inputs(self):
        self.redis_server.mset({"inputs": pickle.dumps(self.mutated_inputs)})
    
    # check if the new DNN model can predict the mutated inputs without triggering error
    def checkFailed(self):
        if self.fit_func == "nan":
            errors = pickle.loads(self.redis_server.hget("errors", self.phi[0]))
            # reset errors
            self.redis_server.hset("errors", self.phi[0], pickle.dumps([]))
            
        if self.fit_func == "inc":
            errors_1 = pickle.loads(self.redis_server.hget("errors", self.phi[0]))
            errors_2 = pickle.loads(self.redis_server.hget("errors", self.phi[1]))
            # reset errors
            self.redis_server.hset("errors", self.phi[0], pickle.dumps([]))
            self.redis_server.hset("errors", self.phi[1], pickle.dumps([]))

            errors = []
            if errors_1 != []:
                errors.append([errors_1, self.phi[0]])
            if errors_2 != []:
                errors.append([errors_2, self.phi[1]])
                
            inc_idx = np.where(np.argmax(self.FFunc.predictions_1, 1)!=np.argmax(self.FFunc.predictions_2, 1))[0]
            if inc_idx.size != 0:
                errors.append(self.muated_inputs[inc_idx])
        
        return errors


def ga_main(ga, r1, r2, mut_rate, m, n, layer_idx, maxIter):
    F = []
    fits = []
    start_time = time.time()
    for i in range(maxIter):
        print("Running at iteration {}:".format(i+1))
        iter_start_time = time.time()
        ga.computeFitness(layer_idx)
        top_k_fitness = ga.getTopK_Fit(m)
        fits.append(top_k_fitness)
        P_prime = ga.select(m)
        test_cases = []
        test_cases.extend(P_prime)
        while len(test_cases) < n:
            ga.selectParents()
            ga.crossover(r1)
            r = random.uniform(0,1)
            if r < r2:
                x_2prime = ga.mutate(mut_rate)
                test_cases.append(x_2prime)

        ga.mutated_inputs = np.array(test_cases)
        ga.update_inputs()
        X = ga.checkFailed()
        if X != []:
            F.extend(X)
        iter_end_time = time.time()
        
        print("top {} fitness scores: ".format(m), top_k_fitness[1])
        print("Taken time: {}".format(iter_end_time-iter_start_time))
        print()
    
    end_time = time.time()
    print("Total taken time: {}".format(end_time-start_time))

    return F, fits