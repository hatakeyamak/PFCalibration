from __future__ import division

from matplotlib.colors import LogNorm
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats
import numpy as np
import math

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression


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

def plot_history(history,hist):
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
