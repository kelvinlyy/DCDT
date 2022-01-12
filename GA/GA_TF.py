import numpy as np
import redis
import random
from FFunc import NanFFunc, InconsistencyFFunc
import pickle

class GA_TF:
    '''
    arguments e.g.:
    fit_func: [fitness_function, arguments:[]]
    fit_func: ["inc", ["theano"]]
    model: model
    dataset: x_test
    init_input_mut: 0.1
    init_weight_mut: 0
    r1 (crossover rate): 0.5
    r2 (mutation rate): 0.1
    m (number of parents): 5
    n (maximum dataset size allowed): 1000
    db_flag: 0
    '''
    def __init__(self, fit, model, dataset, init_input_mut, init_weight_mut, r1, r2, m, n, db_flag):
        self.framework = "tensorflow"

        self.fit_func = fit[0]
        self.fit_arg = fit[1]

        self.model = model
        self.dataset = dataset

        self.init_input_mut = init_input_mut
        self.init_weight_mut = init_weight_mut
        self.r1 = r1
        self.r2 = r2
        self.m = m
        self.n = n

        self.db_flag = db_flag
        self.redis_server = redis.Redis(db=self.db_flag)
        if not self.redis_server.ping():
            raise Exception("Redis server not set up")

        
        
    def initPopulation(self):
        # randomly choose n inputs from the dataset if n < len(dataset)
        if len(self.dataset) > self.n:
            selected_x = self.dataset[np.random.choice(len(self.dataset), size=self.n, replace=False)]
        else:
            selected_x = self.dataset
            
        # get model weights
        original_weights = np.array(self.model.get_weights(), dtype=object)
            
        # initialize variables
        self.mutated_inputs = selected_x
        self.mutated_weights = original_weights

        # input level
        if self.init_input_mut != 0 and self.init_input_mut != None:
            self.mutated_inputs = (np.clip((selected_x/255 + np.random.standard_cauchy(selected_x.shape) * self.init_input_mut),0,1) * 255).astype(int)
            
        # weight level
        if self.init_weight_mut != 0 and self.init_weight_mut != None:
            self.mutated_weights = []
            for layer_weight in original_weights:
                self.mutated_weights.append(layer_weight + np.random.standard_cauchy(layer_weight.shape) * self.init_weight_mut)
            
            self.mutated_weights = np.array(self.mutated_weights)
            self.model.set_weights(self.mutated_weights)
                
        return [self.mutated_inputs, self.mutated_weights]

    def prepareFitness(self):
        if self.fit_func == "inc":
            backends = ["tensorflow", self.fit_arg[0]]
            self.FFunc = InconsistencyFFunc(self.redis_server, self.db_flag, backends, self.model, self.mutated_inputs)
            self.FFunc.prepare()
        elif self.fit_func == "nan":
            backend = "tensorflow"
            self.FFunc = NanFFunc(self.redis_server, self.db_flag, backend, self.model, self.mutated_inputs)
            self.FFunc.prepare()
        

    def computeFitness(self, layer_idx=-1):
        if self.fit_func == "inc":
            self.fitness_values = self.FFunc.compute(layer_idx)
            return self.fitness_values
        elif self.fit_func == "nan":
            self.fitness_values = self.FFunc.compute(layer_idx)
            return self.fitness_values
        
    def getTopK_Fit(self, k):
        topK_idx = np.argpartition(self.fitness_values, -k)[-k:]
        return [self.mutated_inputs[topK_idx], self.fitness_values[topK_idx]]
        
    # select m candidates for parents of next-generation inputs
    def select(self):
        selected_index = np.argpartition(self.fitness_values, -self.m)[-self.m:]
        self.selected_x = self.mutated_inputs[selected_index]
        return self.selected_x
    
    # select 2 parents from the candidates
    def selectParents(self):
        self.x1, self.x2 = random.sample(list(self.selected_x), 2)
        return self.x1, self.x2
    
    # return a flatten list of a crossover product of the selected parents
    def crossover(self):
        x1_flatten = self.x1.flatten()
        x2_flatten = self.x2.flatten()

        x1_factor = np.random.choice(2, size=x1_flatten.shape, p=[1-self.r1, self.r1])
        x2_factor = 1 - x1_factor
        
        self.x_prime = x1_flatten * x1_factor + x2_flatten * x2_factor

        return self.x_prime
    
    # mutate the crossover product and reshape it as the shape of the input instance
    def mutate(self):
        input_shape = self.dataset[0].shape
        self.x_2prime = (np.clip((self.x_prime/255 + np.random.standard_cauchy(self.x_prime.shape) * self.r2),0,1) * 255).astype(int)
        self.x_2prime = self.x_2prime.reshape(input_shape)
        return self.x_2prime
    
    def update_inputs(self):
        self.redis_server.mset({"inputs": pickle.dumps(self.mutated_inputs)})
    
    
    # check if the new DNN model can predict the mutated inputs without triggering error
    def checkFailed(self):
        try:
            if self.fit_func == "inc":
                return []

            elif self.fit_func == "nan":
                predictions = self.model.predict(self.mutated_inputs)
                if np.isnan(predictions).any(): # if there is any nan in the predictions
                    return self.mutated_inputs
        except Exception as e:
            return [e]
            