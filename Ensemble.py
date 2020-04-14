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
      
def get_models(n, t):
    """
    Busca os hiperparâmetros dos modelos escolhidos para compor o ensemble
    :parametro series: base de dados
    :parametro n: tipo do modelo
    :parametro t: número modelos
    :return: lista com o dicionário dos modelos que irão compor o ensemble
    """
    models = []
    if n == 0:
        for i in range(t):
            models.append(basic.random_CNN1())
    elif n == 1:
        for i in range(t):
            models.append(basic.random_CNN2())
    elif n == 2:
        for i in range(t):
            models.append(basic.random_CNN3())
    elif n == 3:
        for i in range(2):
            models.append(basic.random_CNN1())
        for i in range(2):
            models.append(basic.random_CNN2())
        for i in range(2):
            models.append(basic.random_CNN3())
    return models
    
def get_dados(models, series_treino, series_teste):
    #series_treino, series_teste,_ = get_search_dataset()
    series_teste = pd.concat([series_treino[series_treino.shape[0] - models['lags']:], series_teste])
    series_treino = series_treino.values.reshape(-1, 1)
    series_teste = series_teste.values.reshape(-1, 1)
    scaler = MinMaxScaler()
    series_treino = scaler.fit_transform(series_treino)
    series_teste = scaler.fit_transform(series_teste)
  
    return series_treino, series_teste, scaler
    

def executa_models(models, train, test):
    """
    Executa os modelos do ensemble
    :parametro models: modelos
    :parametro dataset: base de dados
    :return: yhat: valor previsto, y_test: valor real
    """
    yhats = []
    for i in range(len(models)):
        series_treino, series_teste, scaler = get_dados(models[i], train, test)
        X_train, y_train, X_test, y_test = basic.slideWindow(series_treino, series_teste, models[i]['lags'])
        if models[i]['tipo'] == 0:
            model, history = basic.modelo_CNN1(X_train, y_train, models[i])
        elif models[i]['tipo'] == 1:
            model, history = basic.modelo_CNN2(X_train, y_train, models[i])
        else:
            model, history = basic.modelo_CNN3(X_train, y_train, models[i])
        rmse, yhat, y_test = basic.predictModel(series_teste, model, 10,  models[i]['lags'], scaler)
        yhats.append(yhat)
    return yhats, y_test
    
def gera_distribuicao(yhats):
    """
    Gera distribuição de probabilidade das previsões dos modelos que compõem o ensemble
    :parametro yhats: valores previstos
    :return: distribuição de probabilidade
    """
    y, ys, yss, kde_l, kde_list = [], [], [], [], []
    for z in range(yhats[0].shape[1]):
        for j in range(yhats[0].shape[0]):
            for i in range(len(yhats)):
                y.append(yhats[i][j,z])
            yy = np.array(y)
            y = []
            ys.append(yy)
        yss.append(ys)
        ys = []

    for i in range(len(yss)):
        for j in range(len(yss[0])):
            band = grid(yss[i][j])
            kde = KernelDensity(kernel='gaussian', bandwidth=band).fit(yss[i][j].reshape(-1, 1))
            kde_l.append(kde)
        kde_list.append(kde_l)
        kde_l = []

    return kde_list
    
def predict_ensemble(kde_list, y_test):
    """
    Faz a previsão do ensemble usando a média de 100 amostras da distribuição de probabilidade
    :parametro kde_list: distribuição de probabilidade
    :parametro y_test: valores reais
    :return: rmse da previsão do ensemble e os valores previstos pelo ensemble
    """
    yhat_e = np.zeros((len(kde_list[0]),len(kde_list)))
    rmse_e = []
    for i in range(len(kde_list)):
        for j in range(len(kde_list[0])):
            yhat_e[j][i] = (kde_list[i][j].sample(100)).mean()
    for i in range(10):
        rmse_e.append(np.sqrt(mean_squared_error(yhat_e[:,i],y_test[:])))
  
    return rmse_e, yhat_e
    
def grid(x):
    """
    GridSearch para escolha da  largura de banda(h) do método KDE
    :parametro x: dados
    :return: largura de banda(h)
    """
    bandwidths = np.linspace(0, 1, 10)
    gd = GridSearchCV(KernelDensity(kernel='gaussian'),{'bandwidth': bandwidths}, cv=len(x))
    x = np.array(x)
    gd.fit(x.reshape(-1, 1))
    return np.round(gd.best_params_['bandwidth'],2)
    
def executa(train, test, tipo_ensemble, num_modelos):
  """
  Executa o ensemble
  :parametro dataset: base dados
  :parametro tipo_ensemble: 0 = ensemble CNN1, 1 = ensemble CNN2, 2 = ensemble CNN3, 3 = ensemble híbrido
  return: rmse da previsão, valores previstos, valores reais e modelos do ensemble
  """
  models = get_models(n=tipo_ensemble, t=num_modelos)
  yhats, y_test = executa_models(models, train, test)
  kde_list = gera_distribuicao(yhats)
  rmse_e, yhat_e = predict_ensemble(kde_list, y_test)

  return rmse_e, yhat_e, kde_list
