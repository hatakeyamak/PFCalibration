from __future__ import division

import plot
import process_data
import train


import pandas as pd
import numpy as np
import math

### Allocate the GPU usage on lpc
train.AllocateVRam()

###################################################################
########### DEFINE inputs and target for training and the ROOT file
inputVariables = ['eta', #'charge']#,'pf_hoRaw'], "p", 'pt'
                  'phi', 
                  'pf_totalRaw','pf_ecalRaw','pf_hcalRaw']

targetVariables = ['gen_e']
inputFiles = ["singlePi_histos_trees_depth_samples.root"]
#inputFiles = ["singlePi_histos_trees_new_samples.root","singlePi_histos_trees_valid.root"]

### Get data from inputTree
dataset, compareData = process_data.Get_tree_data(inputFiles,
                                                  inputVariables, targetVariables,               
                                                  withTracks = False, withDepth = True,
                                                  endcapOnly = False, barrelOnly = False,
                                                  isTrainProbe = True)
               
### prepare test and training data
train_data, test_data, train_labels, test_labels = process_data.PreProcess(dataset, targetVariables)

##############################################################
########## Build and Train the model
test_predictions = train.doMachineLearn(train_data, train_labels, test_data, test_labels)
##############################################################
##########Recover meaningful predictions
test_predictions, test_labels = process_data.PostProcess(test_predictions, test_data, test_labels)

##############################################################
########## TRAINING ANALYSIS 
results = test_data.copy()
results['gen_e'] = test_labels
results['DNN'] = test_predictions


compareData['Response'] = (compareData['pf_totalRaw'] - compareData['gen_e'])/compareData['gen_e']
results['Response'] = (results['DNN']-results['gen_e'])/results['gen_e']


##############################################
########## PLOT MAKING #######################

plot.plot_perf(results, None, "Pred vs True")

plot.plot_hist_compare([compareData['Response'],results['Response']],100,['Raw','Keras'],"(Pred-True)/True [E]","pdf/perf_comparison.pdf")
### compare pt distribution ###
plot.plot_hist_compare([compareData['pf_totalRaw'],results['DNN']],25,['Raw','Keras'],"E","pdf/pt_comparison.pdf")

### Pred/True vs True ###
#plot.profile_plot_compare(compareData['gen_e'], compareData['pf_totalRaw']/compareData['gen_e'], 'Raw',
#                     test_labels, test_predictions/test_labels, 'Keras',
#                     100, 0, 500,
#                     "True [E]", "Pred/True [E]", "scale_comparison.pdf")
### Response vs True ###
plot.profile_plot_compare(compareData['gen_e'], compareData['Response'], 'Raw',
                          results['gen_e'], results['Response'], 'Keras',
                          100, 0, 500,
                          "True [E]", "(Pred-True)/True [E]", "pdf/response_comparison.pdf")

### Handle EH and H seperately ###
plot.profile_plot_compare(compareData['gen_e'][compareData['EH Hadron'] == 1], compareData['Response'][compareData['EH Hadron'] == 1], 'Raw EH Had',
                          results['gen_e'][results['EH Hadron'] == 1], results['Response'][results['EH Hadron']==1], 'Corr EH Had',
                          100, 0, 500,
                          "True [E]", "(Pred-True)/True [E]", "pdf/eh_response_comparison.pdf")

plot.profile_plot_compare(compareData['gen_e'][compareData['H Hadron'] == 1], compareData['Response'][compareData['H Hadron'] == 1], 'Raw H Had',
                          results['gen_e'][results['H Hadron'] == 1], results['Response'][results['H Hadron']==1], 'Corr H Had',
                          100, 0, 500,
                          "True [E]", "(Pred-True)/True [E]", "pdf/h_response_comparison.pdf")

### Response vs Eta ###

plot.profile_plot_compare(abs(compareData['eta']), compareData['Response'], 'Raw',
                          abs(results['eta']), results['Response'], 'Keras',
                          24, 0, 2.4,
                          "Eta", "(Pred-True)/True [E]", "pdf/response_vs_eta.pdf")


plot.EH_vs_E_plot(results['pf_ecalRaw']/results['gen_e'],results['pf_hcalRaw']/results['gen_e'],
                  results['pf_ecalRaw']/results['DNN'], results['pf_hcalRaw']/results['DNN'],
                  50, 'Raw', 'Corrected')

plot.E_bin_response(compareData,results,20,['Raw','Keras'],-1.2,1.2,"(Pred-True)/True (GeV)","pdf/1DResponse.pdf")    

