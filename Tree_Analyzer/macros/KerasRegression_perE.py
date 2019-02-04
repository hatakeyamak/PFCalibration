import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers
print(tf.__version__)

import uproot
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats
import numpy as np
import math

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

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

def build_model(shape):
    model = keras.Sequential([
        layers.Dense(5, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l1(0.000), input_shape=[shape]),
        #keras.layers.Dropout(0.2),
        layers.Dense(10, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.010)),
        #keras.layers.Dropout(0.7),
        layers.Dense(30, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.001)),
        #keras.layers.Dropout(0.5),
        layers.Dense(15, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l1(0.001)),
        #keras.layers.Dropout(0.0),# 7
        layers.Dense(10, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l1(0.01)),
        keras.layers.Dropout(0.7),
        layers.Dense(1)
        ])
  #optimizer = tf.train.RMSPropOptimizer(0.001)
    optimizer  = 'adam'

    model.compile(loss='mae',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model

### Add training and target variables here
inputVariables = ['gen_e','eta','pf_totalRaw','pf_ecalRaw','pf_hcalRaw','pf_hoRaw']
targetVariable = 'gen_e'
###
### Get data from inputTree
inputTree = uproot.open("singlePi_histos_trees_test.root")["t1"]
#print inputTree
dataset = inputTree.pandas.df(inputVariables)
dataset = dataset.dropna()
### to directly compare to Raw
my_cut = abs(dataset['eta'])<1.5
dataset = dataset[my_cut]
dataset = dataset.reset_index()
compareData = dataset.copy()

dataset['pf_ecalRaw'] = dataset['pf_ecalRaw']/dataset['pf_totalRaw']
dataset['pf_hcalRaw'] = dataset['pf_hcalRaw']/dataset['pf_totalRaw']
dataset['pf_hoRaw'] = dataset['pf_hoRaw']/dataset['pf_totalRaw']

minEnergy = 0
maxEnergy = 400
results = pd.DataFrame(columns=['gen_e','DNN'])

for x in range(0,10):
  range_cut = (dataset['pf_totalRaw'] >= x*(maxEnergy-minEnergy)/10) & (dataset['pf_totalRaw'] < (x + 1)*(maxEnergy-minEnergy)/10)
  print  str(x*(maxEnergy-minEnergy)/10)+"< PFE <"+str((x + 1)*(maxEnergy-minEnergy)/10)
  cut_dataset = dataset[range_cut].copy()
##Prepare Data for training
## Mirror low E data across zero to help with performance
#mirror_Data = dataset[dataset['gen_e']<=50].copy()
#mirror_Data['gen_e'] = -mirror_Data['gen_e']
#mirror_Data['pf_totalRaw'] = - mirror_Data['pf_totalRaw']
#dataset = pd.concat([dataset,mirror_Data], ignore_index=True)
###
### define Test and Train Data as well as the target
  train_dataset = cut_dataset.sample(frac=0.9,random_state=1)
#train_dataset = dataset.sample(n=1000, random_state=0)
  test_dataset  = cut_dataset.drop(train_dataset.index)
  test_dataset  = test_dataset[test_dataset['gen_e']>=0]
#print len(train_dataset[train_dataset['gen_e']>=0]), len(train_dataset)

  train_labels = train_dataset.pop(targetVariable)
#train_labels = np.log(train_labels/train_dataset['pf_totalRaw'])###########
#train_labels = train_labels/train_dataset['pf_totalRaw']####################
#train_labels = np.log(train_labels)

  test_labels  = test_dataset.pop(targetVariable)
#test_labels = np.log(test_labels/test_dataset['pf_totalRaw'])#############
#test_labels = test_labels/test_dataset['pf_totalRaw']######################
#test_labels = np.log(test_labels)


###
  train_stats = train_dataset.describe()
  train_stats = train_stats.transpose()
### need to normalize the data due to scaling/range differences in variable set


  normed_train_data = dropIndex(norm(train_dataset))
  normed_test_data = dropIndex(norm(test_dataset))

  scaled_train_data = dropIndex(scale(train_dataset))
  scaled_test_data = dropIndex(scale(test_dataset))

#print scaled_test_data
#print normed_test_data.tail()
#print test_dataset.tail()


############################################
### Build the model

  

  model = build_model(len(scaled_train_data.keys()))

###########################################
### Train the model
  EPOCHS = 300

  history = model.fit(
    scaled_train_data, train_labels,
    epochs=EPOCHS, batch_size=10000,
    validation_data=(scaled_test_data,test_labels),
    verbose=0)
  
  keras.utils.plot_model(model, to_file="model.png", show_shapes=True)

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

### Predict 
  test_predictions = model.predict(scaled_test_data).flatten()

###Recover meaningful predictions
#test_predictions = np.exp(test_predictions)*test_dataset['pf_totalRaw']###############
#test_predictions = test_predictions*test_dataset['pf_totalRaw']#######################
#test_predictions = np.exp(test_predictions)

#test_labels = np.exp(test_labels)*test_dataset['pf_totalRaw']#########################
#test_labels = test_labels*test_dataset['pf_totalRaw']#################################
#test_labels = np.exp(test_labels)

### Remove fake data across zero
  buffer_results = pd.DataFrame({'gen_e':test_labels,'DNN':test_predictions})
  results = results.append(buffer_results, ignore_index=True, sort=False)
  #print results


def plot_perf(results,cut,title):
  results = (results[cut] if str(cut) != 'None' else results)
  plt.scatter(results['gen_e'], results['DNN'], marker='+', label=title)
  plt.xlabel('True Values [E]')
  plt.ylabel('Predictions [E]')
  #plt.title(title)
  plt.grid(True)
  #plt.axis('equal')
  #plt.axis('square')
  plt.xlim([0,425])
  plt.ylim([0,425])
  plt.legend()
  plt.plot([-425, 425], [-425, 425])

fig = plt.figure(figsize=(16, 8))
#plot_perf(results,None,"All")
[plot_perf(results,
           (results['gen_e']>(i*50)) & (results['gen_e']<((i+1)*50)),
           "label[pt]>"+str(i*50)+" & label[pt]<"+str((i+1)*50)) for i in range(0,11)]
linear_regressor = LinearRegression()
linear_regressor.fit(results['gen_e'].values.reshape(-1, 1),results['DNN'].values.reshape(-1, 1))
fit = linear_regressor.predict(results['gen_e'].values.reshape(-1, 1))
mpl.rcParams['agg.path.chunksize'] = 10000
plt.plot(results['gen_e'].values.reshape(-1, 1),fit,'--k')
fig.savefig("model_acc.png")
plt.show()
plt.clf()
plt.close()

###compare performance of the networks
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

### compare resonpse ###
res_DNN   = (compareData['pf_totalRaw']-compareData['gen_e'])/compareData['gen_e']
res_Keras = (results['DNN']-results['gen_e'])/results['gen_e']
plot_hist_compare([res_DNN,res_Keras],100,['Raw','Keras'],"(Pred-True)/True [E]","perf_comparison.pdf")
### compare pt distribution ###
plot_hist_compare([compareData['pf_totalRaw'],results['DNN']],25,['Raw','Keras'],"E","pt_comparison.pdf")

##### testing profile plot in python

def profile_plot_compare(x1,y1,label_1,x2,y2,label_2,bins,xmin,xmax,xlabel,ylabel,fig_name):
  def setupBins(x,y,bins,xmin,xmax):
    means_result = scipy.stats.binned_statistic(x, [y,y**2], bins=bins, range=(xmin,xmax), statistic='mean')
    bin_count = scipy.stats.binned_statistic(x, [y,y**2], bins=bins, range=(xmin,xmax), statistic='count')
    means, means2 = means_result.statistic
    standard_deviations = np.sqrt(means2 - means**2)/np.sqrt(bin_count.statistic)
    bin_edges = means_result.bin_edges
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2.
    return bin_centers, means, standard_deviations

  x1, y1, yerr1 = setupBins(x1, y1, bins, xmin, xmax)
  x2, y2, yerr2 = setupBins(x2, y2, bins, xmin, xmax)

  fig = plt.figure()
  plt.errorbar(x=x1, y=y1, yerr=yerr1, xerr=(xmax-xmin)/(2*bins), linestyle='none', marker='.', label =label_1)
  plt.errorbar(x=x2, y=y2, yerr=yerr2, xerr=(xmax-xmin)/(2*bins), linestyle='none', marker='.', label =label_2)
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
                     results['gen_e'], results['DNN']/results['gen_e'], 'Keras',
                     100, 1, 501,
                     "True [E]", "Pred/True [E]", "scale_comparison.pdf")
### Response vs True ###
profile_plot_compare(compareData['gen_e'], (compareData['pf_totalRaw']-compareData['gen_e'])/compareData['gen_e'], 'Raw',
                     results['gen_e'], (results['DNN']-results['gen_e'])/results['gen_e'], 'Keras',
                     100, 1, 501,
                     "True [E]", "(Pred-True)/True [E]", "response_comparison.pdf")
##### Debug low pt #####
if False:
  for i, label in enumerate(results['DNN']):
    if label < 10:
      print (label, results['DNN'][i])
