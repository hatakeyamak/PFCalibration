from __future__ import division
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers
from tensorflow import set_random_seed
set_random_seed(2)
print(tf.__version__)

import uproot
import pandas as pd
from matplotlib.colors import LogNorm
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats
import numpy as np
import math

from tensorflow.python.keras import backend as K
from keras import backend as k
from tensorflow.python.ops import math_ops
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5
k.tensorflow_backend.set_session(tf.Session(config=config))

def plot_hist_compare(x,bins,labels,xlabel,fig_name):

  fig = plt.figure()
  plt.hist(x, 
           bins,
           density = True,
           histtype = 'step', 
           label = labels)

  plt.xlabel(xlabel)
  plt.grid(True)
  plt.legend()
  plt.show()
  fig.savefig(fig_name)
  plt.clf()
  plt.close()

### Add training and target variables here
inputVariables = ['gen_e','eta', 'phi', 'pf_totalRaw','pf_ecalRaw','pf_hcalRaw', 
                  'pf_hcalFrac1', 'pf_hcalFrac2', 'pf_hcalFrac3', 'pf_hcalFrac4', 'pf_hcalFrac5', 'pf_hcalFrac6', 'pf_hcalFrac7']#'charge']#,'pf_hoRaw'], "p", 'pt'
targetVariable = 'gen_e'
### InputFiles
#inputFiles = ["singlePi_histos_trees_new_samples.root","singlePi_histos_trees_valid.root"]
inputFiles = ["singlePi_histos_trees_depth_samples.root"]
### Get data from inputTree
def TChain(inputFile):
  data = pd.DataFrame()
  for inputFile in inputFiles:
#print inputTree
    inputTree = uproot.open(inputFile)["t1"]
    data = pd.concat([data,inputTree.pandas.df(inputVariables)], ignore_index=True) 
  return data

dataset = TChain(inputFiles)
dataset = dataset.dropna()
#print dataset[dataset['gen_e']>500]
### to directly compare to Raw
my_cut = (abs(dataset['eta'])<1.5) & (dataset['pf_totalRaw'] >0) & (dataset['gen_e']>0) & (dataset['gen_e']<500)
train_cut     = (dataset['pf_totalRaw']-dataset['gen_e'])/dataset['gen_e'] > -0.90 ## dont train on bad data with response of -1 
dataset = dataset[(my_cut) & (train_cut)]
dataset = dataset.reset_index()

dataset['H Hadron']  = ((dataset['pf_ecalRaw'] == 0) & (dataset['pf_hcalRaw'] > 0))*1
dataset['EH Hadron'] = ((dataset['pf_ecalRaw'] > 0) & (dataset['pf_hcalRaw'] == 0))*1


##Prepare Data for training
#dataset['pf_ecalRaw'] = dataset['pf_ecalRaw']/dataset['pf_totalRaw']
#dataset['pf_hcalRaw'] = dataset['pf_hcalRaw']/dataset['pf_totalRaw']
#dataset['pf_hoRaw'] = dataset['pf_hoRaw']/dataset['pf_totalRaw']

## Mirror low E data across zero to help with performance
#mirror_Data = dataset[dataset['gen_e']<=50].copy()
#mirror_Data['gen_e'] = -mirror_Data['gen_e']
#mirror_Data['pf_totalRaw'] = - mirror_Data['pf_totalRaw']
#dataset = pd.concat([dataset,mirror_Data], ignore_index=True)
###
### define Test and Train Data as well as the target
temp_data = dataset[dataset['gen_e'] <= 200]
a = len(dataset[(dataset['gen_e']>200) & (dataset['gen_e']<=400)] )
b = len(dataset[dataset['gen_e']<=200])
temp_data = temp_data.sample(frac=int(a)/int(b),random_state=1)

#print 'tempData', temp_data

dataset = pd.concat([temp_data, dataset[dataset['gen_e']>200]],ignore_index=False)
#dataset = dataset.drop('index', axis=1)
compareData = dataset.copy()
#plot_hist_compare(dataset['gen_e'],25,'test','test','test')

train_dataset = dataset.sample(frac=0.75,random_state=1)
#train_cut     = (train_dataset['pf_totalRaw']-train_dataset['gen_e'])/train_dataset['gen_e'] > -0.9 ## dont train on bad data with response of -1 
#train_dataset = train_dataset[train_cut]
#train_dataset = dataset.sample(n=1000, random_state=0)
test_dataset  = dataset.drop(train_dataset.index)
test_dataset  = test_dataset[test_dataset['gen_e']>=0]
#print len(train_dataset[train_dataset['gen_e']>=0]), len(train_dataset)

train_labels = train_dataset.pop(targetVariable)
#train_labels = np.log(train_dataset['pf_totalRaw']/train_labels)###########
train_labels = (train_dataset['pf_totalRaw']/train_labels)-1####################
#train_labels = np.log(train_labels)

test_labels  = test_dataset.pop(targetVariable)
#test_labels = np.log(test_dataset['pf_totalRaw']/test_labels)#############
test_labels = (test_dataset['pf_totalRaw']/test_labels)-1######################
#test_labels = np.log(test_labels)

train_spect = train_dataset[['H Hadron','EH Hadron']].copy()
del train_dataset['H Hadron'],train_dataset['EH Hadron']
test_spect = test_dataset[['H Hadron','EH Hadron']].copy()
del test_dataset['H Hadron'],test_dataset['EH Hadron']


###
train_stats = train_dataset.describe()
train_stats = train_stats.transpose()
### need to normalize the data due to scaling/range differences in variable set
def norm(data):
  return (data - train_stats['mean']) / train_stats['std']
def scale(data):
  x = data.values
  scaler = preprocessing.StandardScaler().fit(x)
  x_scaled = scaler.transform(x)
  scaledData = pd.DataFrame(x_scaled, columns=data.columns)
  return scaledData.copy()
def dropIndex(data):
  data = data.drop(columns='index')
  return data.copy()

normed_train_data = dropIndex(norm(train_dataset))
normed_test_data = dropIndex(norm(test_dataset))

scaled_train_data = dropIndex(scale(train_dataset))
scaled_test_data = dropIndex(scale(test_dataset))


#train_labels = train_labels/np.amax(train_labels)
#test_max = np.amax(test_labels)
#test_labels  = test_labels/test_max

#print test_labels, train_labels
#print scaled_test_data
#print normed_test_data.tail()
#print test_dataset.tail()


############################################
### Build the model
def build_model():
  model = keras.Sequential([
    layers.Dense(12, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.000), input_shape=[len(scaled_train_data.keys())]),# 4, relu, l1(0)
    #keras.layers.Dropout(0.5), # 0
    layers.Dense(32, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.000)),#16, relu, l2(0)
    keras.layers.Dropout(0.7),# 0.5 
    layers.Dense(4, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l1(0.0000)), #2, relu, l2(0)
    #keras.layers.Dropout(0.5), # 0
    #layers.Dense(16, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l1(0.0000)), #2, relu, l2(0)
    #keras.layers.Dropout(0.5), # 0
    layers.Dense(1)
  ])
  #optimizer = tf.train.RMSPropOptimizer(0.001)
  optimizer  = keras.optimizers.Adam(lr=.0001, decay= .0001 / 300) # lr = .0001, decay = .0001 / 200
  
  def RMSLE(y_true,y_pred):
    first_log = math_ops.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    second_log = math_ops.log(K.clip(y_true, K.epsilon(), None) + 1.)
    return K.sqrt(K.mean(math_ops.square(first_log - second_log), axis=-1))
  
  def MSER(y_true,y_pred):
    return K.mean(K.square(y_pred-y_true)+K.abs(y_pred-y_true),axis=-1)

  model.compile(loss='mse', # mse
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

model = build_model()

###########################################
### Train the model
EPOCHS = 300 # 200

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5) # 10
history = model.fit(
  scaled_train_data, train_labels, # target is the response
  epochs=EPOCHS, batch_size=5128, # 5128
  validation_data=(scaled_test_data,test_labels),
  verbose=1, callbacks=[early_stop])
###########################################
####Analyze the training history  
#keras.utils.plot_model(model, to_file="model.png", show_shapes=True)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch

def plot_history(history):
  fig, [mean, mean_square] = plt.subplots(1, 2, figsize=(12, 6))
  mean.set_xlabel('Epoch')
  mean.set_ylabel('Mean Abs Error [E]')
  mean.grid(True)
  mean.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  mean.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  #mean.set_ylim(top=.3)
  mean.set_yscale('log')
  mean.legend()
  
  mean_square.set_xlabel('Epoch')
  mean_square.set_ylabel('Mean Square Error [$E^2$]')
  mean_square.grid(True)
  mean_square.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  mean_square.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  mean_square.set_yscale('log')
  mean_square.legend()
  
  plt.show()
  fig.savefig("mean_Drop.pdf")
  plt.clf()
  plt.close()

plot_history(history)

###################################
loss, mae, mse = model.evaluate(scaled_test_data, test_labels, verbose=1)

print("Testing set Mean Abs Error: {:5.2f} E".format(mae))
###################################
### Predict using test Dataset
### (reverse feature prep here) 
test_predictions = model.predict(scaled_test_data).flatten()

###Recover meaningful predictions
#test_predictions = 1/(np.exp(test_predictions))*test_dataset['pf_totalRaw']###############
test_predictions = 1/((test_predictions+1)/test_dataset['pf_totalRaw'])#######################
#test_predictions = np.exp(test_predictions)
#test_predictions = test_predictions*test_max

#test_labels = 1/(np.exp(test_labels))*test_dataset['pf_totalRaw']#########################
test_labels = 1/((test_labels+1)/test_dataset['pf_totalRaw'])#################################
#test_labels = np.exp(test_labels)
#test_labels = test_labels*test_max

###################################
results = pd.DataFrame({'gen_e':test_labels,'DNN':test_predictions})

### analysis of the performance ###
def plot_perf(results, cut,title):
  results = (results[cut] if str(cut) != 'None' else results)
  plt.hist2d(results['gen_e'], results['DNN'], bins=100, norm=LogNorm(), label=title)
  plt.xlabel('True Values [E]')
  plt.ylabel('Predictions [E]')
  #plt.title(title)
  plt.grid(True)
  #plt.axis('equal')
  #plt.axis('square')
  plt.xlim([0,500])
  plt.ylim([0,500])
  plt.colorbar()
  plt.legend()
  plt.plot([-500, 500], [-500, 500])

fig = plt.figure(figsize=(16, 8))
#plot_perf(results,None,"All")
plot_perf(results, None, "Pred vs True")
#[plot_perf(results,
#           (results['gen_e']>(i*50)) & (results['gen_e']<((i+1)*50)),
#           "label[pt]>"+str(i*50)+" & label[pt]<"+str((i+1)*50)) for i in range(0,11)]
def lin_regressor(x,y):
  linear_regressor = LinearRegression()
  linear_regressor.fit(x.values.reshape(-1, 1),y.values.reshape(-1, 1))
  fit = linear_regressor.predict(x.values.reshape(-1, 1))
  mpl.rcParams['agg.path.chunksize'] = 10000
  plt.plot(x.values.reshape(-1, 1),fit,'--k')
lin_regressor(results['gen_e'][results['gen_e']<=200],results['DNN'][results['gen_e']<=200])
lin_regressor(results['gen_e'][(results['gen_e']<=500) & (results['gen_e']>200)],results['DNN'][(results['gen_e']<=500) & (results['gen_e']>200)])
fig.savefig("model_acc.png")
plt.show()
plt.clf()
plt.close()

###compare performance of the networks

### compare resonpse ###
res_DNN   = (compareData['pf_totalRaw']-compareData['gen_e'])/compareData['gen_e']
res_Keras = (test_predictions-test_labels)/test_labels
plot_hist_compare([res_DNN,res_Keras],100,['Raw','Keras'],"(Pred-True)/True [E]","perf_comparison.pdf")
### compare pt distribution ###
plot_hist_compare([compareData['pf_totalRaw'],test_predictions],25,['Raw','Keras'],"E","pt_comparison.pdf")

##### testing profile plot in python

def profile_plot_compare(x1,y1,label_1,x2,y2,label_2,bins,xmin,xmax,xlabel,ylabel,fig_name):
  pred_x = x2
  pred_y = y2
  def gausMean(x,y,bins,xmin,xmax):
    df = pd.DataFrame({'x':x,'y':y})
    x_bins = np.linspace(xmin,xmax,bins+1)
    df['bin'] = np.digitize(x,bins=x_bins)
    binned = df.groupby('bin')
    temp_df = pd.DataFrame()
    for i_bin in range(1,bins+1):
      temp_list = binned.get_group(i_bin)
      temp_list = temp_list.assign(mean=temp_list.y.mean(), std=temp_list.y.std())
      temp_list = temp_list[(temp_list['y'] >= temp_list['mean'] - temp_list['std']) & (temp_list['y'] <= temp_list['mean'] + temp_list['std'])]
      if i_bin == 1:
        temp_df = temp_list
      else:
        temp_df = temp_df.append(temp_list, ignore_index=True)
    return temp_df['x'], temp_df['y']
    
  def setupBins(x,y,bins,xmin,xmax):
    means_result = scipy.stats.binned_statistic(x, [y,y**2], bins=bins, range=(xmin,xmax), statistic='mean')
    bin_count = scipy.stats.binned_statistic(x, [y,y**2], bins=bins, range=(xmin,xmax), statistic='count')
    means, means2 = means_result.statistic
    standard_deviations = np.sqrt(means2 - means**2)/np.sqrt(bin_count.statistic)
    bin_edges = means_result.bin_edges
    bin_width = (bin_edges[1] - bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width/2
    return bin_centers, means, standard_deviations
    
  x1, y1 = gausMean(x1, y1, bins, xmin, xmax) # cut out tails
  x2, y2 = gausMean(x2, y2, bins, xmin, xmax) # cut out tails
  x1, y1, yerr1 = setupBins(x1, y1, bins, xmin, xmax)
  x2, y2, yerr2 = setupBins(x2, y2, bins, xmin, xmax)
  fig = plt.figure()
  plt.errorbar(x=x1, y=y1, yerr=yerr1, xerr=(xmax-xmin)/(2*bins), linestyle='none', marker='.', label =label_1)
  plt.errorbar(x=x2, y=y2, yerr=yerr2, xerr=(xmax-xmin)/(2*bins), linestyle='none', marker='.', label =label_2)
  plt.hist2d(pred_x, pred_y, bins=bins, norm=LogNorm(), range=np.array([(xmin,xmax),(-0.5,1.5)]), label=label_2)
  plt.colorbar()
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.grid(True)
  plt.legend()
  plt.show()
  fig.savefig(fig_name)
  plt.clf()
  plt.close()

### Pred/True vs True ###
profile_plot_compare(compareData['gen_e'], compareData['pf_totalRaw']/compareData['gen_e'], 'Raw',
                     test_labels, test_predictions/test_labels, 'Keras',
                     100, 0, 500,
                     "True [E]", "Pred/True [E]", "scale_comparison.pdf")
### Response vs True ###
profile_plot_compare(compareData['gen_e'], (compareData['pf_totalRaw']-compareData['gen_e'])/compareData['gen_e'], 'Raw',
                     test_labels, (test_predictions-test_labels)/test_labels, 'Keras',
                     100, 0, 500,
                     "True [E]", "(Pred-True)/True [E]", "response_comparison.pdf")


def EH_vs_E_plot(raw_Efrac, raw_Hfrac, corr_Efrac, corr_Hfrac, bins, label_raw, label_corr):
  fig, [raw, corr] = plt.subplots(1, 2, figsize=(12, 6))
  [temp.set_xlabel('H/T (GeV)') for temp in [raw,  corr]]
  [temp.set_ylabel('E/T (GeV)') for temp in [raw,  corr]]
  [temp.grid(True) for temp in [raw,  corr]]

  raw_ = raw.hist2d(x=raw_Hfrac, y=raw_Efrac, bins=bins, norm=mpl.colors.LogNorm(vmin=1,vmax=1*10E5)
, range=np.array([(0,1.5),(0,1.5)]))
  raw.set_title(label_raw)
  corr_ = corr.hist2d(x=corr_Hfrac, y=corr_Efrac, bins=bins,  norm=mpl.colors.LogNorm(vmin=1,vmax=1*10E5), range=np.array([(0,1.5),(0,1.5)]))
  corr.set_title(label_corr)
  fig.colorbar(corr_[3])
  plt.show()
  fig.savefig("EH_vs_E.pdf")
  plt.clf()
  plt.close()

EH_vs_E_plot(test_dataset['pf_ecalRaw']/test_labels,test_dataset['pf_hcalRaw']/test_labels,
             test_dataset['pf_ecalRaw']/test_predictions, test_dataset['pf_hcalRaw']/test_predictions,
             50, 'Raw', 'Corrected')

#def E_bin_response():
##### Debug low pt #####
if False:
  for i, label in enumerate(test_labels):
    if label < 10:
      print (label, test_predictions[i])
