import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import random
import math
from operator import itemgetter
import statsmodels

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
import hickle as hkl

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

from CNN import basic

class Ensemble_CNN:
  def __init__(self, type_models, number):
    self.type_models = type_models
    self.number = number
    self.models = []
    self.trained_models = []
  pass

  def properties(self):
    print('Models types:', self.type_models)
    print('Number models:', self.number)
    print('Models:', self.models)
    print('Trained Models:', self.trained_models)

  def generate_models(self):
    """
    Search the hyperparameters of the models chosen to compose the ensemble

    """
    if self.type_models == 'CNN1':
        for i in range(self.number):
            self.models.append(basic.random_CNN1())
    elif self.type_models == 'CNN2':
        for i in range(self.number):
            self.models.append(basic.random_CNN2())
    elif self.type_models == 'CNN3':
        for i in range(self.number):
            self.models.append(basic.random_CNN3())
    elif self.type_models == 'CNNH':
        for i in range(self.number):
            self.models.append(basic.random_CNN1())
        for i in range(self.number):
            self.models.append(basic.random_CNN2())
        for i in range(self.number):
            self.models.append(basic.random_CNN3())
    pass
  
  def get_test(self, data, max_lag, i):
    """
    Pre-processing of data
    :parameter train: training database
    :parameter test: test database
    """
    test = data[max_lag-self.models[i]['lags']:]
    return test
  
  def slideWindow(self, data, n_lags):
    """
    Separates input and output data
    parameter data: data
    parameter n_lags: number of lags used by the model
    """
    X = []
    y = []
    for i in range(n_lags, len(data)):
        X.append(data[i-n_lags:i])
        y.append(data[i])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X,(X.shape[0],X.shape[1],1))
    return X, y

  def fit(self, data):
    """
    Run the ensemble models
    :parameter train: training database
    :parameter test: test database
    """
    yhats = []
    for i in range(len(self.models)):
        X_train, y_train = self.slideWindow(data, self.models[i]['lags'])
        if self.models[i]['tipo'] == 0:
            model, history = basic.modelo_CNN1(X_train, y_train, self.models[i])
        elif self.models[i]['tipo'] == 1:
            model, history = basic.modelo_CNN2(X_train, y_train, self.models[i])
        else:
            model, history = basic.modelo_CNN3(X_train, y_train, self.models[i])
        self.trained_models.append(model)
    pass

  def get_maxlag(self):
    l = []
    for i in range(len(self.models)):
      l.append(self.models[i]['lags'])
    max_lag = max(l)
    return max_lag

  def point_forecast(self, data, forecast_horizon):
    """
    Point forecast of the ensemble
    :parameter test: test database
    :parameter forecast_horizon: number of forecasts ahead
    """
    yhats = []
    max_lag = self.get_maxlag()
    data = self.get_test(data, max_lag, 0)
    for i in range(len(self.trained_models)):
      test = self.get_test(data, max_lag, i)
      X_test, y_test = self.slideWindow(test, self.models[i]['lags'])
      yhat,_ = self.predict_models(test, forecast_horizon, i)
      if i==0:
        mean = yhat
      mean = np.mean( np.array([mean, yhat]), axis=0)
      yhats.append(yhat)
    return mean, y_test, yhats
       
  def predict_models(self, test, forecast_horizon, z):
    """
    Point forecast of the individual models
    :parameter test: test database
    :parameter forecast_horizon: number of forecasts ahead
    """
    X_test, y_test = self.slideWindow(test, self.models[z]['lags'])
    yhat = np.zeros((y_test.shape[0],forecast_horizon))
    for i in range(len(X_test)):
        X = X_test[i,:,0].reshape((1, X_test.shape[1], X_test.shape[2]))
        for j in range(forecast_horizon):
            yhat[i,j] = self.trained_models[z].predict(X, verbose=0)
            X = np.insert(X,self.models[z]['lags'],yhat[i,j],axis=1) 
            X = np.delete(X,0,axis=1)
    return yhat, y_test#, rmse

  def probabilistic_forecast(self, data, forecast_horizon):
    """
    Gera distribuição de probabilidade das previsões dos modelos que compõem o ensemble
    :parametro yhats: valores previstos
    :return: distribuição de probabilidade
    """
    mean, y_test, yhats = self.point_forecast(data, forecast_horizon)
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
            band = self.grid(yss[i][j])
            kde = KernelDensity(kernel='gaussian', bandwidth=band).fit(yss[i][j].reshape(-1, 1))
            kde_l.append(kde)
        kde_list.append(kde_l)
        kde_l = []

    return kde_list, y_test
  
  def grid(self, x):
    """
    GridSearch for choosing the bandwidth (h) of the KDE method
    :parametro data: data
    :return: bandwidth(h)
    """
    bandwidths = np.linspace(0, 1, 10)
    gd = GridSearchCV(KernelDensity(kernel='gaussian'),{'bandwidth': bandwidths}, cv=len(x))
    x = np.array(x)
    gd.fit(x.reshape(-1, 1))
    return np.round(gd.best_params_['bandwidth'],2)

  @staticmethod 
  def load_ensemble(name):
    import hickle as hkl
    m = hkl.load(name)
    obj = Ensemble_CNN(m.type_models, m.number)
    obj.models = m.models
    obj.trained_models = m.trained_models
    return obj
  
  def save_ensemble(self, file, file_name):
    hkl.dump(file, file_name)
    pass

  def generate_distribution(yhats):
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

  def metrics(self, yhat, y_test, forecast_horizon):
    rmse = []
    mae = []
    mape = []
    for i in range(forecast_horizon):
        rmse.append(np.sqrt(mean_squared_error(yhat[:,i],y_test[:])))
        mae.append(mean_absolute_error(yhat[:,i],y_test[:]))
        mape.append(np.mean(np.abs((yhat[:,i] - y_test[:]) / y_test[:])) * 100)
    return rmse, mae, mape
