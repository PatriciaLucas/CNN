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
from itertools import repeat

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

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
from scipy import stats
import seaborn
from CNN import basic
import os
import re


def return_labels(matriz_path):
  lista = []
  labels = []
  #matriz_path = '/content/gdrive/MyDrive/Dados/Matriz20x150'
  matrizPaths = os.listdir(matriz_path) #nome dos arquivos
  for matrix in matrizPaths: #Exemplo: '10-15Maca_3.npy'
    label = ''.join(i for i in matrix if not i.isdigit()) #Exemplo: '-Maca_.npy'
    label = ''.join(c for c in label if c not in '-') #Exemplo: 'Maca_.npy'
    label = ''.join(c for c in label if c not in '_') #Exemplo: 'Maca.npy'
    label = label.replace('.npy', '') #Exemplo: 'Maca'
    labels.append(label) #adiciona o item no final da lista
    lista.append(matrix)

  lb = LabelBinarizer()
  dados_Y = lb.fit_transform(labels)
  indices = range(len(labels))
  return indices, dados_Y, lista

def generate_train(train_X, matriz_path, lista):
    """
    Gera dados de treino para a CNN4
    :parametro train_X: dados de entrada de treino
    :return: dados de entrada e saída de treino
    """
    data = []
    labels = []
    idx=train_X
    matriz_treino = map(lambda i: lista[i], idx)
    matriz_treino = sorted(matriz_treino, key=lambda x: (int(re.sub('\D','',x)),x))
    #matriz_path = 'gdrive/My Drive/Dados/Matriz20x150/'

    for matriz in matriz_treino:
      mat = np.load(matriz_path + '/' + matriz)
      #mat = scaler.fit_transform(mat)
      #mat = stats.zscore(mat)
      one_channel = np.stack((mat,)*1, axis=-1)
      label = ''.join(i for i in matriz if not i.isdigit()) #Exemplo: '-Maca_.npy'
      label = ''.join(c for c in label if c not in '-') #Exemplo: 'Maca_.npy'
      label = ''.join(c for c in label if c not in '_') #Exemplo: 'Maca.npy'
      label = label.replace('.npy', '') #Exemplo: 'Maca'
      labels.append(label)
      data.append(one_channel)
          
    x_train = np.array(data, dtype = 'float32')
    lb = LabelBinarizer()
    y_train = lb.fit_transform(labels)
    y_train = np.stack((y_train,)*1, axis=-1)
    return x_train, y_train

def generate_test(test_X, matriz_path, lista):
    """
    Gera dados de teste para a CNN4
    :parametro test_X: dados de entrada de teste
    :return: dados de entrada e saída de teste
    """
    data = []
    labels = []
    idx=test_X
    matriz_treino = map(lambda i: lista[i], idx)
    matriz_treino = sorted(matriz_treino, key=lambda x: (int(re.sub('\D','',x)),x))
    #matriz_path = 'gdrive/My Drive/Dados/Matriz20x150/'

    for matriz in matriz_treino:
      mat = np.load(matriz_path + '/' + matriz)
      #mat = scaler.fit_transform(mat)
      #mat = stats.zscore(mat)
      one_channel_video = np.stack((mat,)*1, axis=-1)
      label = ''.join(i for i in matriz if not i.isdigit()) #Exemplo: '-Maca_.npy'
      label = ''.join(c for c in label if c not in '-') #Exemplo: 'Maca_.npy'
      label = ''.join(c for c in label if c not in '_') #Exemplo: 'Maca.npy'
      label = label.replace('.npy', '') #Exemplo: 'Maca'
      labels.append(label)
      data.append(one_channel_video)
          
    x_test = np.array(data, dtype = 'float32')
    lb = LabelBinarizer()
    y_test = lb.fit_transform(labels)
    y_test = np.stack((y_test,)*1, axis=-1)
    return x_test, y_test

def initial_population(n, cnn):
    """
    Gera a população
    :parametro n: número de indivíduos da população
    :return: população
    """
    pop = []
    for i in range(n):
        if cnn == 'CNN1':
            pop.append(random_CNN1())
        elif cnn == 'CNN2':
            pop.append(random_CNN2())
        else:
            pop.append(random_CNN3())
    return pop

def random_CNN1(): 
  return genotypeCNN1(
    random.randint(0, 2),   # número de filtros [16,32,64,128]
    random.randint(0, 2),   # probabilidade de pooling [0%, 50%, 100%]
    random.randint(2, 5), #tamanho da janela de pooling [2,3,4,5] 
    random.uniform(0.5, 0.8),  # porcentagem dropout
    random.randint(0, 1),   # normalização (0 - sim, 1 - não)
    random.randint(2, 200),   # quantidade de lags
    random.randint(1, 6), # número de camadas convolucionais
    random.randint(0, 3), # tamanho do kernel de convolução
    [], #RMSE de teste do modelo
    []  #número de parâmetros do modelo
  )

def random_CNN2():
  num_conv = random.randint(1, 6) # número de camadas convolucionais
  k = random.randint(0, 3) # tamanho do kernel de convolução
  
  if k == 0: 
    kernel_size = 2
  elif k == 1:
    kernel_size = 3
  elif k == 2:
    kernel_size = 5
  else:
    kernel_size = 11
    
  max_lags = (2**num_conv)*kernel_size
  min_lags = (2**num_conv)
  
  return genotypeCNN2(
    random.randint(0, 2),   # número de filtros [16,32,64,128]
    random.uniform(0, 0.5),  # porcentagem de dropout [0.5..0.8]
    random.randint(0, 1),   # normalização (0 - sim, 1 - não)
    random.randint(min_lags, max_lags),   # quantidade de lags
    num_conv,
    kernel_size,
    [], #RMSE de teste do modelo
    []  #número de parâmetros do modelo
  )

def random_CNN3():
  num_conv = random.randint(1, 6) # número de camadas convolucionais
  k = random.randint(0, 3) # tamanho do kernel de convolução
  pilhas = random.randint(1, 3) #número de blocos [1,2,3]
  if k == 0: 
    kernel_size = 2
  elif k == 1:
    kernel_size = 3
  elif k == 2:
    kernel_size = 5
  else:
    kernel_size = 11
    
  max_lags = ((2**num_conv)*kernel_size)*pilhas
  min_lags = (2**num_conv)*kernel_size
  
  return genotypeCNN3(
    pilhas,
    random.randint(0, 2),   # número de filtros [16,32,64]
    random.uniform(0, 0.5),  # porcentagem dropout
    random.randint(0, 1),   # normalização (0 - sim, 1 - não)
    random.randint(min_lags, max_lags),   # quantidade de lags
    num_conv,
    kernel_size,
    [], #RMSE de teste do modelo
    []  #número de parâmetros do modelo
  )

def genotypeCNN1(filters, pool, pool_size, dropout, norm, lags, num_conv, kernel_size, rmse, num_param):
  ind = {
      'filters': filters, # número de filtros [16,32,64]
      'pool': pool, # probabilidade de pooling [0%, 50%, 100%] 
      'pool_size': pool_size, #tamanho da janela de pooling [2,3,4,5] 
      'dropout': dropout, # porcentagem dropout (0 a 0.5)
      'norm': norm, # normalização (0 - sim, 1 - não)
      'lags': lags, # quantidade de lags [2 - 200]
      'num_conv': num_conv, # número de camadas convolucionais [1 - 6]
      'kernel_size': kernel_size, # tamanho do kernel de convolução [2,3,5,11]
      'rmse': rmse, #RMSE de teste do modelo
      'num_param':num_param   #número de parâmetros do modelo

  }
  return ind

def genotypeCNN2(filters, dropout, norm, lags, num_conv, kernel_size, rmse, num_param):
  ind = {
      'filters': filters, # número de filtros [16,32,64]
      'dropout': dropout, # porcentagem de dropout [0.5..0.8]
      'norm': norm, # normalização (0 - sim, 1 - não)
      'lags': lags, # quantidade de lags em relação ao número de convoluções e tamanho do kernel
      'num_conv': num_conv, # número de camadas convolucionais
      'kernel_size': kernel_size, # tamanho do kernel de convolução [2,3,5,11]
      'rmse': rmse, #RMSE de teste do modelo
      'num_param':num_param   #número de parâmetros do modelo

  }
  return ind

def genotypeCNN3(blocos, filters, dropout, norm, lags, num_conv, kernel_size, rmse, num_param):
  ind = {
      'pilhas': blocos, #número de blocos [1,2,3]
      'filters': filters, # número de filtros [16,32,64]
      'dropout': dropout, # porcentagem dropout [0.5...0.8]
      'norm': norm, # normalização (0 - sim, 1 - não)
      'lags': lags, # quantidade de lags em relação ao número de convoluções e tamanho do kernel
      'num_conv': num_conv, # número de camadas convolucionais
      'kernel_size': kernel_size, # tamanho do kernel de convolução [2,3,5,11]
      'rmse': rmse, #RMSE de teste do modelo
      'num_param':num_param   #número de parâmetros do modelo

  }
  return ind

def modelo_CNN1(X_train, y_train, X_test, y_test, individual, epocas):
    """
    Cria um modelo CNN1
    :parametro X_train: dados para treinamento
    :parametro y_train: rótulo dos dados de treinamento
    :parametro individual: dicionário com os hiperparâmetros do modelo
    :return: o modelo
    """
    warnings.filterwarnings('ignore')
    call = [EarlyStopping(monitor='loss', mode='min', patience=15, verbose=1),]
    model = Sequential()
  
    if individual['filters'] == 0: 
        filters = 16
    elif individual['filters'] == 1:
        filters = 32
    else:
        filters = 64
  
    if individual['kernel_size'] == 0: 
        kernel_size = 2
    elif individual['kernel_size'] == 1:
        kernel_size = 3
    elif individual['kernel_size'] == 2:
        kernel_size = 5
    else:
        kernel_size = 11
    
    try:
        for i in range(individual['num_conv']):
            model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=(X_train.shape[1],1),
                                padding='same', kernel_constraint=max_norm(3)))
            if individual['pool'] == 0: 
                model.add(MaxPooling1D(pool_size=individual['pool_size'], strides=2, padding='same', data_format='channels_first'))
            elif individual['pool'] == 1:
                rnd = random.uniform(0,1)
                if rnd > .5:
                    model.add(MaxPooling1D(pool_size=individual['pool_size']))
            model.add(SpatialDropout1D(round(individual['dropout'],1)))
            if individual['norm'] == 0:
                model.add(BatchNormalization())

        model.add(Flatten())
        model.add(Dense(1, activation = 'linear'))
        model.compile(loss='mse', optimizer='Adam')
        history = model.fit(X_train, y_train, epochs = epocas, verbose=0, batch_size = filters, validation_data=(X_test, y_test), callbacks = call)
        
        return model, history
    except Exception as ex:
        individual = basic.random_CNN1()
        model, history = modelo_CNN1(X_train, y_train, X_test, y_test, individual,epocas)
        return model, history

def modelo_CNN2(X_train, y_train, X_test, y_test, individual, epocas):
    """
    Cria um modelo CNN2
    :parametro X_train: dados para treinamento
    :parametro y_train: rótulo dos dados de treinamento
    :parametro individual: dicionário com os hiperparâmetros do modelo
    :return: o modelo
    """
    warnings.filterwarnings('ignore')
    model = Sequential()
    call = [EarlyStopping(monitor='loss', mode='min', patience=15, verbose=1),]
    if individual['filters'] == 0: 
        filters = 16
    elif individual['filters'] == 1:
        filters = 32
    else:
        filters = 64
  
    if individual['kernel_size'] == 0: 
        kernel_size = 2
    elif individual['kernel_size'] == 1:
        kernel_size = 3
    elif individual['kernel_size'] == 2:
        kernel_size = 5
    else:
        kernel_size = 11
    
    for i in range(individual['num_conv']):
        d = 2**i
        model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=(X_train.shape[1],1),
                             padding='causal',dilation_rate = d, kernel_constraint=max_norm(3)))
        model.add(SpatialDropout1D(round(individual['dropout'],1)))
        if individual['norm'] == 0:
            model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='Adam')
    history = model.fit(X_train, y_train, epochs = epocas, verbose=0, batch_size = filters, validation_data=(X_test, y_test), callbacks = call)          
    return model, history

def modelo_CNN3(X_train, y_train, X_test, y_test, individual,epocas):
    """
    Cria um modelo CNN3
    :parametro X_train: dados para treinamento
    :parametro y_train: rótulo dos dados de treinamento
    :parametro individual: dicionário com os hiperparâmetros do modelo
    :return: o modelo
    """
    warnings.filterwarnings('ignore')
    call = [EarlyStopping(monitor='loss', mode='min', patience=15, verbose=1),]
    if individual['filters'] == 0: 
        filters = 16
    elif individual['filters'] == 1:
        filters = 32
    else:
        filters = 64
    
    if individual['norm'] == 0:
        norm = False
    else:
        norm = True
  
    if individual['kernel_size'] == 0: 
        kernel_size = 2
    elif individual['kernel_size'] == 1:
        kernel_size = 3
    elif individual['kernel_size'] == 2:
        kernel_size = 5
    else:
        kernel_size = 11
    
    d = []
    for i in range(individual['num_conv']):
        d.append(2**i)
    i = Input(batch_shape=(None,X_train.shape[1],1))
    o = TCN(nb_filters=filters, kernel_size=kernel_size, nb_stacks=individual['pilhas'], dilations=d,
    padding='causal', use_skip_connections=False, dropout_rate=individual['dropout'], return_sequences=False, name='tcn')(i)
        
    o = Dense(1)(o)
    model = Model(inputs=[i], outputs=[o])
    model.compile(optimizer='Adam', loss='mse')  
    history = model.fit(X_train, y_train, epochs = epocas, verbose=0, batch_size = filters, validation_data=(X_test, y_test), callbacks = call)      
  
    return model, history

def modelo_CNN4(X_train, y_train, X_test, y_test, individual,epocas):
    """
    Cria um modelo CNN4
    :parametro X_train: dados para treinamento
    :parametro y_train: rótulo dos dados de treinamento
    :parametro individual: dicionário com os hiperparâmetros do modelo
    :return: o modelo
    """
    warnings.filterwarnings('ignore')
    call = [EarlyStopping(monitor='loss', mode='min', patience=15, verbose=1),]
    if individual['filters'] == 0: 
        filters = 16
    elif individual['filters'] == 1:
        filters = 32
    else:
        filters = 64
    
    if individual['norm'] == 0:
        norm = False
    else:
        norm = True
  
    if individual['kernel_size'] == 0: 
        kernel_size = 2
    elif individual['kernel_size'] == 1:
        kernel_size = 3
    elif individual['kernel_size'] == 2:
        kernel_size = 5
    else:
        kernel_size = 11
    
    d = []
    for i in range(individual['num_conv']):
        d.append(2**i)
    
    model = compiled_tcn(return_sequences=False, num_feat=150, num_classes=20, nb_filters=filters, kernel_size=kernel_size, dilations=d,
                         padding='causal', dropout_rate=individual['dropout'], use_batch_norm=True, nb_stacks=individual['pilhas'], max_len=X_train[0:1].shape[1],
                         use_skip_connections=True)
    y_train = y_train.squeeze().argmax(axis=1)
    y_test = y_test.squeeze().argmax(axis=1)
    history = model.fit(X_train, y_train, epochs=epocas,validation_data=(X_test, y_test), workers=4, use_multiprocessing=True, callbacks = call, verbose=0)    
  
    return model, history

def evaluation(individual, cnn, series, epocas):
    """
    Avalia os indivíduos da população
    :parametro individual: indivíduo da população
    :parametro cnn: tipo da rede
    :parametro series: base de dados
    :return: número de parâmetros do modelo e a média do rmse
    """ 
    if individual['filters'] == 0: 
        filters = 16
    elif individual['filters'] == 1:
        filters = 32
    else:
        filters = 64

    results = []
    
    if cnn == 'CNN4':
      indices, dados_Y, lista = return_labels(series)
      for k in range (0, 3):
        # 75% treino - 25% teste
        (train_X, test_X, train_Y, test_Y) = train_test_split(indices,dados_Y,random_state=42,test_size=0.25,stratify=dados_Y)
        X_train, y_train = generate_train(train_X, series, lista)
        X_test, y_test = generate_test(test_X, series, lista)
        model, history  = modelo_CNN4(X_train, y_train, X_test, y_test, individual, epocas)
        results.append(np.sqrt(history.history['val_loss'][-1]))
        print(results)
    else:
      windows_size=.5
      train_size=.7
      w = int(len(series) * windows_size)
      d = int(.2 * w)
      X_train = []
      y_train = []
      X_test = []
      y_test = []
      i=0
      while i < w:
          train_index = (series[i:i+int(w*train_size)].index.values.astype(int))
          test_index = (series[i+int(w*train_size):w+i].index.values.astype(int))
          train = series[train_index].values
          test = series[test_index].values
          X_train, y_train, X_test, y_test = basic.slideWindow(train, test, individual['lags'])
          
          if cnn == 'CNN1':
              model, history  = modelo_CNN1(X_train, y_train, X_test, y_test, individual, epocas)
          elif cnn == 'CNN2':
              model, history  = modelo_CNN2(X_train, y_train, X_test, y_test, individual, epocas)
          else:
              model, history  = modelo_CNN3(X_train, y_train, X_test, y_test, individual, epocas)
          results.append(np.sqrt(history.history['val_accuracy'][-1]))
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
    filho = genotypeCNN1(filters, pool, pool_size, dropout, norm, lags, num_conv, kernel_size, rmse, num_param)

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
    
    filho = genotypeCNN2(filters, dropout, norm, lags, num_conv, kernel_size, rmse, num_param)
  
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
    pilhas = best['pilhas'] if rnd < .7 else worst['pilhas']
  
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
    lags = random.randint((2**num_conv)*k,(2**num_conv)*k*pilhas)
    
    rmse = []
    num_param = []
    
    filho = genotypeCNN3(pilhas, filters, dropout, norm, lags, num_conv, kernel_size, rmse, num_param)
  
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
    individual['pilhas'] = min(2, max(1,int(individual['pilhas'] + np.random.normal(0,1))))
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
    individual['lags'] = random.randint((2**individual['num_conv'])*kernel_size,(2**individual['num_conv'])*kernel_size*individual['pilhas'])
  
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

def genetic(ngen, npop, pcruz, pmut, dataset, cnn, epocas):
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
    populacao = initial_population(npop, cnn)
    melhor_rmse = []
    media_rmse = []
    melhor_len_lags = []
    media_len_lags = []

    res = list(map(evaluation, populacao, repeat(cnn), repeat(dataset), repeat(epocas)))
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
      
        res = list(map(evaluation, populacao, repeat(cnn), repeat(dataset), repeat(epocas)))
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
        if cnn == 'CNN4':
          h1, = pl.plot(melhor_rmse, c='blue', label='Best ACC')
          h2, = pl.plot(media_rmse, c='cyan', label='Mean ACC')
          pl.title("ACC")
        else:
          h1, = pl.plot(melhor_rmse, c='blue', label='Best RMSE')
          h2, = pl.plot(media_rmse, c='cyan', label='Mean RMSE')
          pl.title("RMSE")
          pl.legend([h1, h2],['Best','Mean'])

        pl.subplot(122)
        h3, = pl.plot(melhor_len_lags, c='red', label='Best Número de parâmetros')
        h4, = pl.plot(media_len_lags, c='orange', label='Mean Número de parâmetros')
        pl.title("Número de parâmetros")
        pl.legend([h3, h4],['Best','Mean'])

        display.clear_output(wait=True)
        display.display(pl.gcf())

    melhorT = sorted(populacao, key=lambda item: item['rmse'])[0]

    return melhorT
