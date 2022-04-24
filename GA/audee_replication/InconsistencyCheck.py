from GA_audee import GA, ga_main
import numpy as np

# GA hyperparameters
mut_level = 'i'
init_noise = 0 # scale of Cauchy noise added in initialization
r1 = 0.5 # crossover rate
r2 = 0.3 # mutation rate
r3 = 0.01 # mutation scale
m = 3 # top m chromosomes selected
n = 4 # population size
layer_idx = -1 # index of layer used in calculation of fitness values
maxIter = 3


def ga_inc(backend_1, backend_2, model, x, input_scale, db_flag):
    fit = ['inc', [backend_1, backend_2]]
    ga = ga_main(fit, mut_level, model, x, input_scale, init_noise, r1, r2, r3, m, n, layer_idx, db_flag, maxIter)
    formatted_P = ga.formatPopulations(ga.P)
    x_max = formatted_P[np.argmax(ga.fit_hist[-1])]
    return x_max