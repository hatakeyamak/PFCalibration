from __future__ import division
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers
from tensorflow import set_random_seed
set_random_seed(2)
print(tf.__version__)

import plot

import pandas as pd
import numpy as np

from tensorflow.python.keras import backend as K
from keras import backend as k
from tensorflow.python.ops import math_ops
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

def AllocateVRam():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    k.tensorflow_backend.set_session(tf.Session(config=config))


def Build_Model(N_inputs, N_epochs):
    model = keras.Sequential([
        layers.Dense(N_inputs, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.01), input_shape=[N_inputs]),# 4, relu, l1(0)
    #keras.layers.Dropout(0.5), # 0
        layers.Dense(28, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.00)),#16, relu, l2(0)
        keras.layers.Dropout(0.0),# 0.5 
        layers.Dense(28, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.00)), #2, relu, l2(0)
        keras.layers.Dropout(0.0), # 0
        layers.Dense(28, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.00)), #2, relu, l2(0)
        keras.layers.Dropout(0.0), # 0
        layers.Dense(28, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.00)), #2, relu, l2(0)
        keras.layers.Dropout(0.0), # 0
        layers.Dense(28, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.00)), #2, relu, l2(0)
        keras.layers.Dropout(0.0), # 0
        layers.Dense(28, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.00)), #2, relu, l2(0)
        keras.layers.Dropout(0.0), # 0
        layers.Dense(28, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.00)), #2, relu, l2(0)
        keras.layers.Dropout(0.0), # 0
        layers.Dense(28, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.00)), #2, relu, l2(0)
        keras.layers.Dropout(0.0), # 0
        layers.Dense(28, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.00)), #2, relu, l2(0)
        keras.layers.Dropout(0.0), # 0
        layers.Dense(28, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.00)), #2, relu, l2(0)
        keras.layers.Dropout(0.0), # 0
        layers.Dense(28, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.00)), #2, relu, l2(0)
        keras.layers.Dropout(0.0), # 0
        layers.Dense(1)
    ])
    #optimizer = tf.train.RMSPropOptimizer(0.001)
    optimizer  = keras.optimizers.Adam(lr=.0001, decay= .0001 / N_epochs) # lr = .0001, decay = .0001 / 200
  
    def RMSLE(y_true,y_pred):
        first_log = math_ops.log(K.clip(y_pred, K.epsilon(), None) + 1.)
        second_log = math_ops.log(K.clip(y_true, K.epsilon(), None) + 1.)
        return K.sqrt(K.mean(math_ops.square(first_log - second_log), axis=-1))
  
    def MSER(y_true,y_pred):
        return K.mean(K.square(y_pred-y_true)+K.abs(y_pred-y_true),axis=-1)

    def WMSER(y_true,y_pred):
        return K.mean(K.abs(K.pow(y_pred-y_true,2)/y_pred), axis=-1)

    model.compile(loss='mae', # mse
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model

def doMachineLearn(train_data, train_labels, test_data, test_labels):


    
    ### Normalie Data
    train_stats = train_data.describe()
    train_stats = train_stats.transpose()
    def norm(data):
        return (data - train_stats['mean']) / train_stats['std']
    def dropIndex(data):
        data = data.drop(columns='index')
        return data.copy()

    train_data = dropIndex(norm(train_data))
    test_data  = dropIndex(norm(test_data))
    print "Training Variables: " + str(train_data.keys())
    print "Target Variables: " + "gen_e" ####### FIX ME #########
    #### Train Model

    EPOCHS = 500;
    model = Build_Model(len(train_data.keys()), EPOCHS)

    #early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20) # 10
    history = model.fit(
        train_data, train_labels, # target is the response
        epochs=EPOCHS, batch_size=5128, # 5128
        validation_data=(test_data,test_labels),
        verbose=1) #callbacks=[early_stop])

    ####Analyze the training history  
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    plot.plot_history(history,hist)
    
    loss, mae, mse = model.evaluate(test_data, test_labels, verbose=1)
    print("Testing set Mean Abs Error: {:5.3f} E".format(mae))

    test_predictions = model.predict(test_data).flatten()
    return test_predictions
