import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import random
import math
from operator import itemgetter
import statsmodels
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, Model, Input
from keras.constraints import max_norm, unit_norm
from keras.layers import Dense, Flatten, SpatialDropout1D, Activation, Add, BatchNormalization, Conv1D, MaxPooling1D
from keras import regularizers
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping
from tcn import TCN
from tcn import compiled_tcn

import pylab as pl
from IPython import display
from matplotlib import pyplot as plt
import seaborn
from CNN import basic

def initial_population(n):
    """
    Gera a população
    :parametro n: número de indivíduos da população
    :return: população
    """
    pop = []
    for i in range(n): 
        pop.append(random_genotype())
    return pop
    
def evaluation(individual, cnn, series):
    """
    Avalia os indivíduos da população
    :parametro individual: indivíduo da população
    :parametro cnn: tipo da rede
    :parametro series: base de dados
    :return: número de parâmetros do modelo e a média do rmse
    """
    windows_size=.5
    train_size=.7
    w = int(len(series) * windows_size)
    d = int(.2 * w)
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    i=0

  
    if individual['filters'] == 0: 
        filters = 16
    elif individual['filters'] == 1:
        filters = 32
    else:
        filters = 64

    results = []
    
    while i < w:
        train_index = (series[i:i+int(w*train_size)].index.values.astype(int))
        test_index = (series[i+int(w*train_size):w+i].index.values.astype(int))
        train = series[train_index].values
        test = series[test_index].values
        X_train, y_train, X_test, y_test = slideWindow(train, test, individual['lags'])
        
        if cnn == 'CNN1':
            model, history  = basic.modelo_CNN1(X_train, y_train, X_test, y_test, individual)
        elif cnn == 'CNN2':
            model, history  = basic.modelo_CNN2(X_train, y_train, X_test, y_test, individual)
        else:
            model, history  = basic.modelo_CNN3(X_train, y_train, X_test, y_test, individual)

        results.append(np.sqrt(history.history['val_loss'][-1]))
        i = i+d

    rmse = np.nanmean(results)
    num_param = model.count_params()
      
    return num_param, rmse
    
def tournament(population, objective):
    """
    Seleção de indivíduos por torneio duplo passo 2
    """
    n = len(population)-1

    r1 = random.randint(0,n) if n > 2 else 0
    r2 = random.randint(0,n) if n > 2 else 1
    ix = r1 if population[r1][objective] < population[r2][objective] else r2
    return population[ix]
  

def selection(population):
    """
    Seleção de indivíduos por torneio duplo passo 1
    """
    pai1 = tournament(population, 'rmse')
    pai2 = tournament(population, 'rmse')

    finalista = tournament([pai1, pai2], 'num_param')

    return finalista
    
def crossover_CNN1(pais):
    """
    Cruzamento
    :parametro pais: lista com dois indivíduos
    :return: individuo filho
    """
    if pais[0]['rmse'] < pais[1]['rmse'] :
        best = pais[0] 
        worst = pais[1]
    else:
        best = pais[1]
        worst = pais[0]
  
    dropout = float(.7*best['dropout'] + .3*worst['dropout'])

    rnd = random.uniform(0,1)
    norm = best['norm'] if rnd < .7 else worst['norm']

    rnd = random.uniform(0,1)
    pool = best['pool'] if rnd < .7 else worst['pool']

    rnd = random.uniform(0,1)
    pool_size = best['pool_size'] if rnd < .7 else worst['pool_size']

    rnd = random.uniform(0,1)
    filters = best['filters'] if rnd < .7 else worst['filters']

    rnd = random.uniform(0,1)
    num_conv = best['num_conv'] if rnd < .7 else worst['num_conv']

    rnd = random.uniform(0,1)
    kernel_size = best['kernel_size'] if rnd < .7 else worst['kernel_size']

    rnd = random.uniform(0,1)
    lags = best['lags'] if rnd < .7 else worst['lags']

    rmse = []
    num_param = []

    filho = basic.genotypeCNN1(filters, pool, pool_size, dropout, norm, lags, num_conv, kernel_size, rmse, num_param)

    return filho
    
def crossover_CNN2(pais):
    """
    Cruzamento
    :parametro pais: lista com dois indivíduos
    :return: individuo filho
    """
    if pais[0]['rmse'] < pais[1]['rmse'] :
        best = pais[0] 
        worst = pais[1]
    else:
        best = pais[1]
        worst = pais[0]
  
    dropout = float(.7*best['dropout'] + .3*worst['dropout'])
  
    rnd = random.uniform(0,1)
    norm = best['norm'] if rnd < .7 else worst['norm']
  
    rnd = random.uniform(0,1)
    filters = best['filters'] if rnd < .7 else worst['filters']
  
    rnd = random.uniform(0,1)
    num_conv = best['num_conv'] if rnd < .7 else worst['num_conv']
    
    rnd = random.uniform(0,1)
    kernel_size = best['kernel_size'] if rnd < .7 else worst['kernel_size']
  
    if kernel_size == 0: 
        k = 2
    elif kernel_size == 1:
        k = 3
    elif kernel_size == 2:
        k = 5
    else:
        k = 11
    lags = random.randint(2**num_conv,(2**num_conv)*k)
    
    rmse = []
    num_param = []
    
    filho = genotype(filters, dropout, norm, lags, num_conv, kernel_size, rmse, num_param)
  
    return filho

def crossover_CNN3(pais):
    """
    Cruzamento
    :parametro pais: lista com dois indivíduos
    :return: individuo filho
    """
    if pais[0]['rmse'] < pais[1]['rmse'] :
        best = pais[0] 
        worst = pais[1]
    else:
        best = pais[1]
        worst = pais[0]
  
    dropout = float(.7*best['dropout'] + .3*worst['dropout'])
  
    rnd = random.uniform(0,1)
    blocos = best['blocos'] if rnd < .7 else worst['blocos']
  
    rnd = random.uniform(0,1)
    norm = best['norm'] if rnd < .7 else worst['norm']
  
    rnd = random.uniform(0,1)
    filters = best['filters'] if rnd < .7 else worst['filters']
  
    rnd = random.uniform(0,1)
    num_conv = best['num_conv'] if rnd < .7 else worst['num_conv']
    
    rnd = random.uniform(0,1)
    kernel_size = best['kernel_size'] if rnd < .7 else worst['kernel_size']
  
    if kernel_size == 0: 
        k = 2
    elif kernel_size == 1:
        k = 3
    elif kernel_size == 2:
        k = 5
    else:
        k = 11
    lags = random.randint((2**num_conv)*k,(2**num_conv)*k*blocos)
    
    rmse = []
    num_param = []
    
    filho = genotype(blocos, filters, dropout, norm, lags, num_conv, kernel_size, rmse, num_param)
  
    return filho
    
def mutation_CNN1(individual):
    """
    Mutação
    :parametro individual: indivíduo que sofrerá a mutação
    :return: individuo mutado
    """
    individual['dropout'] = min(0.8, max(0.5,individual['dropout'] + np.random.normal(0,.1)))
    individual['pool'] = random.randint(0,2)
    individual['pool_size'] = min(5, max(2,int(individual['pool_size'] + np.random.normal(0,1))))
    individual['num_conv'] = min(5, max(1,int(individual['num_conv'] + np.random.normal(0,1))))
    individual['norm'] = random.randint(0,1)
    individual['filters'] = random.randint(0,2)
    individual['lags'] = min(50, max(1,int(individual['lags'] + np.random.normal(0,2))))
    individual['kernel_size'] = random.randint(0,3)

    return individual

def mutation_CNN2(individual):
    """
    Mutação
    :parametro individual: indivíduo que sofrerá a mutação
    :return: individuo mutado
    """
    individual['dropout'] = min(0.8, max(0.5,individual['dropout'] + np.random.normal(0,.1)))
    individual['num_conv'] = min(5, max(1,int(individual['num_conv'] + np.random.normal(0,1))))
    individual['norm'] = random.randint(0,1)
    individual['filters'] = random.randint(0,2)
    individual['kernel_size'] = random.randint(0,3)
    if individual['kernel_size'] == 0: 
        kernel_size = 2
    elif individual['kernel_size'] == 1:
        kernel_size = 3
    elif individual['kernel_size'] == 2:
        kernel_size = 5
    else:
        kernel_size = 11
    a = 2**individual['num_conv']
    b = (2**individual['num_conv'])*kernel_size
    individual['lags'] = random.randint(a,b)
    return individual

def mutation_CNN3(individual):
    """
    Mutação
    :parametro individual: indivíduo que sofrerá a mutação
    :return: individuo mutado
    """
    individual['blocos'] = min(2, max(1,int(individual['blocos'] + np.random.normal(0,1))))
    individual['dropout'] = min(1, max(0,individual['dropout'] + np.random.normal(0,.1)))
    individual['num_conv'] = min(5, max(1,int(individual['num_conv'] + np.random.normal(0,1))))
    individual['norm'] = random.randint(0,1)
    individual['filters'] = random.randint(0,2)
    individual['kernel_size'] = random.randint(0,3)
    if individual['kernel_size'] == 0: 
        kernel_size = 2
    elif individual['kernel_size'] == 1:
        kernel_size = 3
    elif individual['kernel_size'] == 2:
        kernel_size = 5
    else:
        kernel_size = 11
    individual['lags'] = random.randint((2**individual['num_conv'])*kernel_size,(2**individual['num_conv'])*kernel_size*individual['blocos'])
  
    return individual
    
def elitism(population, new_population):
    """
    Inseri o melhor indivíduo da população na nova população e exclui o pior
    """
    population = sorted(population, key=itemgetter('rmse')) 
    best = population[0]

    new_population = sorted(new_population, key=itemgetter('rmse')) 
    new_population[-1] = best

    new_population = sorted(new_population, key=itemgetter('rmse')) 

    return new_population

def genetic(ngen, npop, pcruz, pmut, dataset, cnn):
    """
    Executa o AG
    :parametro ngen: número de gerações
    :parametro npop: número de indivíduos da população
    :parametro pcruz: probabilidade de cruzamento
    :parametro pmut: probabilidade de mutação
    :parametro cnn: string para escolha do tipo de rede ('CNN1', 'CNN2', 'CNN3')
    """
    fig = pl.gcf()
    fig.set_size_inches(15, 5)
    fig, ax = plt.subplots(nrows=1, ncols=1,figsize=[15,5])
    new_populacao = []
    populacao = initial_population(npop)
    melhor_rmse = []
    media_rmse = []
    melhor_len_lags = []
    media_len_lags = []

    res = list(map(evaluation, populacao, repeat(cnn), repeat(dataset)))
    for i in range(len(res)):
        populacao[i]['num_param'],populacao[i]['rmse'] = res[i]

    for i in range(ngen):
        for j in range(int(npop/2)):
            pais = []
            pais.append(selection(populacao))
            pais.append(selection(populacao))

            rnd1 = random.uniform(0,1)
            rnd2 = random.uniform(0,1)
            if cnn == 'CNN1':
                filho1 = crossover_CNN1(pais) if pcruz > rnd1 else pais[0]
                filho2 = crossover_CNN1(pais) if pcruz > rnd2 else pais[1]
            elif cnn == 'CNN2':
                filho1 = crossover_CNN2(pais) if pcruz > rnd1 else pais[0]
                filho2 = crossover_CNN2(pais) if pcruz > rnd2 else pais[1]
            else:
                filho1 = crossover_CNN3(pais) if pcruz > rnd1 else pais[0]
                filho2 = crossover_CNN3(pais) if pcruz > rnd2 else pais[1]
            

            rnd1 = random.uniform(0,1)
            rnd2 = random.uniform(0,1)
            if cnn == 'CNN1':
                filho11 = mutation_CNN1(filho1) if pmut > rnd1 else filho1
                filho22 = mutation_CNN1(filho2) if pmut > rnd2 else filho2
            elif cnn == 'CNN2':
                filho11 = mutation_CNN2(filho1) if pmut > rnd1 else filho1
                filho22 = mutation_CNN2(filho2) if pmut > rnd2 else filho2
            else:
                filho11 = mutation_CNN3(filho1) if pmut > rnd1 else filho1
                filho22 = mutation_CNN3(filho2) if pmut > rnd2 else filho2

            new_populacao.append(filho11)
            new_populacao.append(filho22)
      
        res = list(map(evaluation, populacao, repeat(cnn), repeat(dataset)))
        for i in range(len(res)):
            new_populacao[i]['num_param'],new_populacao[i]['rmse'] = res[i]

        populacao = elitism(populacao, new_populacao)
        _best = populacao[0]

        melhor_rmse.append(_best['rmse'])
        media_rmse.append(sum([k['rmse'] for k in populacao])/len(populacao))
        melhor_len_lags.append(_best['num_param'])
        media_len_lags.append(sum([k['num_param'] for k in populacao])/len(populacao))

        new_populacao = []
    
        pl.subplot(121)
        h1, = pl.plot(melhor_rmse, c='blue', label='Best RMSE')
        h2, = pl.plot(media_rmse, c='cyan', label='Mean RMSE')
        pl.title("RMSE")
        pl.legend([h1, h2],['Best','Mean'])

        pl.subplot(122)
        h3, = pl.plot(melhor_len_lags, c='red', label='Best Número de parâmetros')
        h4, = pl.plot(media_len_lags, c='orange', label='Mean Número de parâmetros')
        pl.title("Número de parâmetros")
        pl.legend([h3, h4],['Best','Mean'])

        #display.clear_output(wait=True)
        display.display(pl.gcf())

    melhorT = sorted(populacao, key=lambda item: item['rmse'])[0]

    return melhorT
