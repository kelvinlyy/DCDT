import cv2
import keras
import numpy as np
import random
import time
import warnings
from Util.Cov_Util import extractModelArchitect, buildModelByArtchitect, modelReduction, total_lines, calc_coverage
import tensorflow as tf
import pandas as pd

layer_dict=pd.read_csv('Util/layer_dict.csv', index_col=0)

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

class Cov_GA:
    def __init__(self, model_architecture = ['Conv2D', 'Flatten', 'Dense'], seedlist = [[0,0,0,0], [], [0]], x=x_test[0]):
        #population: list of (model_architecture, hyperparamters seed list)
        assert len(model_architecture) == len(seedlist), 'layer not matching seed length'
        self.P = []
        
        self.source_architect = model_architecture
        self.source_seedlist = seedlist
        self.source_model_coverage = None
        
        if len(x.shape) == 3:
            self.input = x[None,:]
        else: self.input = x
        
        self.fit_hist = []
        self.fail_P = []
        
    # initialzie model architecture list
    def initPopulation(self, n):
        self.n = n        
        self.P.append((self.source_architect, self.source_seedlist))        
        self.source_model_coverage = calc_coverage(self.source_architect, self.source_seedlist, self.input)
                
        for _ in range(n-1):
            #mutate
            mutate_seedlist = []
            for seed in self.source_seedlist:
                noise = np.array([_ for _ in np.random.standard_cauchy(len(seed))], dtype=int)
                mutate_seedlist.append((seed + noise).tolist())
            self.P.append((self.source_architect, mutate_seedlist))
        
        return 

    # compute fitness scores for populations
    def computeFitness(self):
        Fit = []
        count = 1
        for p in (self.P):
            count+=1
            try:
                Fit.append(total_lines(calc_coverage(p[0], p[1], self.input)))
            except:
                Fit.append(0)
                self.fail_P.append(p)
            
        return Fit

    # select m candidates for parents of next-generation inputs
    def select(self, m, Fit, file=None):
        selected_index = np.argpartition(Fit, -m)[-m:]
        P_prime = [self.P[_] for _ in selected_index]
        if file!=None:
            file.write(f'{P_prime}, ')
            file.write(f'{[Fit[_] for _ in selected_index]}\n')
    
        return P_prime

    # select 2 parents from the candidates
    def selectParents(self, P_prime):
        x1, x2 = random.sample(list(P_prime), 2)
        return x1, x2

    # return a flatten list of a crossover product of the selected parents
    def crossover(self, x1, x2, r):
        a1, sl1 = x1
        a2, sl2 = x2
        COa = []
        COsl = []

        for i in range(max(len(a1),len(a2))):
            if i+1 > len(a1):
                COa.append(a2[i])
                COsl.append(sl2[i])

            elif i+1 > len(a2):
                COa.append(a1[i])
                COsl.append(sl1[i])

            elif random.random() > r:
                COa.append(a2[i])
                COsl.append(sl2[i])
            else:        
                COa.append(a1[i])
                COsl.append(sl1[i])
                
        return (COa, COsl)

    # mutate the crossover product
    def mutate(self, x_prime):
        architecture = x_prime[0][:]
        seedlist = x_prime[1][:]
        
        #add random layer
        aval_layer = []
        for layer in layer_dict.index.tolist():
            if layer not in architecture: aval_layer.append(layer)

        if aval_layer:
            newlayer = random.choice(aval_layer)
            newseed = np.ones(layer_dict.loc[newlayer]['para_num'], dtype=int).tolist()

            insert = int(random.random()*len(architecture))

            architecture.insert(insert, newlayer)
            seedlist.insert(insert, newseed)
        
        #mutate hyperparameters
        mutate_seedlist = []
        for seed in seedlist:
            noise = np.array([_ for _ in np.random.standard_cauchy(len(seed))], dtype=int)
            mutate_seedlist.append((seed + noise).tolist())
        return (architecture, mutate_seedlist)

def cov_ga_main(model_architecture, seedlist, x, m, n, r, maxIter, ga=None):
    sstart_time = time.time()
    
    if ga == None:
        ga = Cov_GA(model_architecture, seedlist, x)
        ga.initPopulation(n)
        
    else:
        print('Continuing from the previous populations...')
        
    prev_iter = len(ga.fit_hist)
    for i in range(prev_iter, prev_iter + maxIter):
        start_time = time.time()
        Fit = ga.computeFitness()
        ga.fit_hist.append(Fit)
        
        if i < prev_iter + maxIter - 1: # no mutation and crossover in the last iteration
            P_prime = ga.select(m, Fit)
            P_pp = []
            P_pp.extend(P_prime)
            while len(P_pp) < n:
                x1, x2 = ga.selectParents(P_prime)
                x_prime = ga.crossover(x1, x2, r)
                #crossover
                x_pp = ga.mutate(x_prime)
                P_pp.append(x_pp)
        ga.P = P_pp
            
        end_time = time.time()
        print('Average fitness value: {}'.format(np.mean(Fit)))
        print('Time taken: {}'.format(end_time - start_time))
        print()
    
    print()
    print('Total time taken:', end_time - sstart_time)
    
    return ga