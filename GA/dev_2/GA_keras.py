from tkinter import E
import keras
import redis
import pickle
import numpy as np
import random
import time
from FFunc_keras import InconsistencyFFunc, NanFFunc
import os

# input should be in range [0,1)
# populations: in flattened shape
# formatted_populations: in normal shape

class GA:
    def __init__(self, fit, mut_level, model, x, input_scale, db_flag):
        self.fit_func, self.backends = fit # a tuple: (fitness function, list of DL framework(s) to be used)
        self.mut_level = mut_level # level: {'i', 'w', 'i+w'}
        
        new_model = keras.models.clone_model(model)
        new_model.set_weights(model.get_weights()) # clone_model does not clone the weights
        self.model = new_model
        
        self.input = x
        self.input_scale = input_scale # {1, 255}
        self.db_flag = db_flag
        
        self.fit_hist = []
        self.F = []
        
        self.redis_server = redis.Redis(db=db_flag)
        if not self.redis_server.ping():
            raise Exception("Redis server not set up")

    # flatten the populations
    def flattenPopulations(self, P):
        flattened_P = []
        for p in P:
            flattened_P.append(np.hstack([i.ravel() for i in p]))

        return np.array(flattened_P)

    # convert the flattened populations to the represented formats
    def formatPopulations(self, P):
        if self.mut_level == 'i':
            input_shape = self.input.shape
            formatted_P = []
            for p in P:
                formatted_P.append(p.reshape(input_shape))

        elif self.mut_level == 'w':
            model_weights = self.model.get_weights()
            formatted_P = []
            for p in P:
                i = 0
                population_weights = []
                for w in model_weights:
                    population_weights.append(p[i:i+np.prod(w.shape)].reshape(w.shape))
                    i += np.prod(w.shape)
                formatted_P.append(population_weights)

        elif self.mut_level == 'i+w':
            input_shape = self.input.shape
            model_weights = self.model.get_weights()
            formatted_P = []
            for p in P:
                i = np.prod(input_shape)
                population_input = p[:i].reshape(input_shape)
                population_weights = []
                for w in model_weights:
                    population_weights.append(p[i:i+np.prod(w.shape)].reshape(w.shape))
                    i += np.prod(w.shape)
                formatted_P.append([np.array(population_input), np.array(population_weights, dtype=object)])

        return np.array(formatted_P, dtype=object)
    
    # initialzie and flatten populations
    def initPopulation(self, init_noise, n):
        self.n = n
        if self.mut_level == 'i':
            chromosome = self.flattenPopulations([self.input])
            self.P = [chromosome]

            for _ in range(n-1):
                noise = np.random.standard_cauchy(chromosome.shape) * init_noise
                self.P.append(np.clip(chromosome + noise, 0, 1))

        elif self.mut_level == 'w':
            model_weights = self.model.get_weights()
            chromosome = self.flattenPopulations([model_weights])
            self.P = [chromosome]

            for _ in range(n-1):
                noise = np.random.standard_cauchy(chromosome.shape) * init_noise
                self.P.append(chromosome + noise)

        elif self.mut_level == 'i+w':
            model_weights = self.model.get_weights()
            chromosome = np.hstack([self.flattenPopulations([self.input]), self.flattenPopulations([model_weights])])
            self.P = [chromosome]

            for _ in range(n-1):
                noise = np.random.standard_cauchy(chromosome.shape) * init_noise
                self.P.append(chromosome + noise)

        else:
            raise Exception("Level parameter only accepts {'i', 'w', 'i+w'}.")

        self.P = np.concatenate(self.P)
        return self.P
    
    # upload the models and inputs to the redis server and prepare the in-class Fitness Function
    def prepareFitness(self, P): # self.model weights may be changed after this
        if self.fit_func == 'nan':
            FFunc = NanFFunc
        elif self.fit_func == 'inc':
            FFunc = InconsistencyFFunc

        formatted_P = self.formatPopulations(P)

        if self.mut_level == 'i':
            self.FFunc = FFunc(self.redis_server, self.db_flag, self.mut_level, self.backends, self.model, None, formatted_P * 255)
        elif self.mut_level == 'w':
            self.FFunc = FFunc(self.redis_server, self.db_flag, self.mut_level, self.backends, self.model, formatted_P, [self.input])
        elif self.mut_level == 'i+w':
            i_P = np.stack(formatted_P[:,0]) * 255# input population
            w_P = np.stack(formatted_P[:,1]) # weight population
            self.FFunc = FFunc(self.redis_server, self.db_flag, self.mut_level, self.backends, self.model, w_P, i_P)

        self.FFunc.prepare()

    # compute fitness scores for each input
    def computeFitness(self, layer_idx):
        Fit = np.squeeze(self.FFunc.compute(layer_idx))
        return Fit

    # select m candidates for parents of next-generation inputs
    def select(self, m, Fit):
        selected_index = np.argpartition(Fit, -m)[-m:]
        P_prime = self.P[selected_index]
        return P_prime

    # select 2 parents from the candidates
    def selectParents(self, P_prime):
        x1, x2 = random.sample(list(P_prime), 2)
        return x1, x2

    # return a flatten list of a crossover product of the selected parents
    def crossover(self, x1, x2, r1):
        x1_factor = np.random.choice(2, size=x1.shape, p=[1-r1, r1])
        x2_factor = 1 - x1_factor
        
        x_prime = x1 * x1_factor + x2 * x2_factor
        return x_prime

    # mutate the crossover product and reshape it as the shape of the input instance
    def mutate(self, x_prime, r2, r3):
        if self.mut_level == 'i':
            x_pp = np.clip(x_prime + np.random.standard_cauchy(x_prime.shape) * np.random.choice(2, x_prime.shape, p=[1-r2, r2]) * r3, 0, 1)
        elif self.mut_level == 'w' or self.mut_level == 'i+w':
            x_pp = x_prime + np.random.standard_cauchy(x_prime.shape) * np.random.choice(2, x_prime.shape, p=[1-r2, r2]) * r3
        
        return x_pp

    # only check nan and crash error
    def checkFailed(self): # check the populations from the previous iteration
        F = []
        if self.fit_func == 'nan':
            with self.redis_server.pipeline() as pipe:
                for i in range(self.n):
                    pipe.hget(f"errors_{i}", self.backends[0])
                errors = pipe.execute()
    
        elif self.fit_func == 'inc':
            with self.redis_server.pipeline() as pipe:
                for i in range(self.n):
                    pipe.hget(f"errors_{i}", self.backends[0])
                    pipe.hget(f"errors_{i}", self.backends[1])
                errors = pipe.execute()

        for e in errors:
            e = pickle.loads(e)
            if e != []:
                F.append([e, self.P])

        return F



def ga_main(fit, mut_level, model, x, input_scale, init_noise, r1, r2, r3, m, n, layer_idx, db_flag, maxIter, ga=None):
    if ga == None:
        ga = GA(fit, mut_level, model, x, input_scale, db_flag)
        ga.initPopulation(init_noise, n)
    else:
        print('Continuing from the previous populations...')
        print()

    for i in range(len(ga.fit_hist), len(ga.fit_hist) + maxIter):
        print(f'Running at iteration {i+1}:')
        start_time = time.time()
        ga.prepareFitness(ga.P)
        Fit = ga.computeFitness(layer_idx)
        ga.fit_hist.append(Fit)

        P_prime = ga.select(m, Fit)
        P_pp = []
        P_pp.extend(P_prime)
        while len(P_pp) < n:
            x1, x2 = ga.selectParents(P_prime)
            x_prime = ga.crossover(x1, x2, r1)
            x_pp = ga.mutate(x_prime, r2, r3)
            P_pp.append(x_pp)

        X = ga.checkFailed()
        if X != []:
            ga.F.append(X)

        ga.P = np.array(P_pp)
        end_time = time.time()

        print('Average fitness value: {}'.format(np.mean(Fit)))
        print('Time taken: {}'.format(end_time - start_time))
        print()
    return ga
        