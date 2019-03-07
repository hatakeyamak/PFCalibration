import uproot
import pandas as pd
import numpy as np 
import math 

def Get_tree_data(inputFiles,inputVariables, targetVariables,
                  withTracks,withDepth,endcapOnly,barrelOnly,
                  isTrainProbe):

#'charge']#,'pf_hoRaw']
    if(withTracks):
        inputVariables += ["p","pt"]
    if(withDepth):
        inputVariables += ['pf_hcalFrac1', 'pf_hcalFrac2', 'pf_hcalFrac3', 
                  'pf_hcalFrac4', 'pf_hcalFrac5', 'pf_hcalFrac6', 'pf_hcalFrac7']

    def TChain(inputFiles):
        data = pd.DataFrame()
        for inputFile in inputFiles:
            print "Opening file: "+str(inputFile)
            inputTree = uproot.open("root/"+inputFile)["t1"]
            data = pd.concat([data,inputTree.pandas.df(inputVariables+targetVariables)], ignore_index=True) 
        return data

    dataset = TChain(inputFiles)
    dataset = dataset.dropna()



    my_cut = (abs(dataset['eta'])< 2.4) & (dataset['pf_totalRaw'] >0) & (dataset['gen_e']>0) & (dataset['gen_e']<500)
    train_cut     = (dataset['pf_totalRaw']-dataset['gen_e'])/dataset['gen_e'] > -0.90 ## dont train on bad data with response of -1 
    dataset = dataset[(my_cut) & (train_cut)]
    if (endcapOnly):
        endcap_train = (abs(dataset['eta'])< 2.4) & (abs(dataset['eta'])>1.6)
        dataset = dataset[endcap_train]
    if (barrelOnly):
        barrel_train = (abs(dataset['eta'])<1.5)
        dataset = dataset[barrel_train]
    dataset = dataset.reset_index()

    dataset['H Hadron']  = ((dataset['pf_ecalRaw'] == 0) & (dataset['pf_hcalRaw'] > 0))*1
    dataset['EH Hadron'] = ((dataset['pf_ecalRaw'] > 0) & (dataset['pf_hcalRaw'] > 0))*1

    if (isTrainProbe):
        dataset = dataset.sample(frac=.25, random_state=1)
    
    compareData = dataset.copy()

    return dataset, compareData

def PreProcess(dataset, targetVariables):

    train_dataset = dataset.sample(frac=0.75,random_state=1)
    test_dataset  = dataset.drop(train_dataset.index)
    test_dataset  = test_dataset[test_dataset['gen_e']>=0]
    
    train_labels = train_dataset.pop(targetVariables[0]) ####### FIX ME #######
    test_labels  = test_dataset.pop(targetVariables[0])  ####### FIX ME #######
    #train_labels = (train_dataset['pf_totalRaw']/train_labels)-1
#test_labels = np.log(test_dataset['pf_totalRaw']/test_labels)#############
    #test_labels = (test_dataset['pf_totalRaw']/test_labels)-1######################

    return train_dataset, test_dataset, train_labels, test_labels

def PostProcess(test_predictions, test_data, test_labels):
    #test_predictions = 1/(np.exp(test_predictions))*test_dataset['pf_totalRaw']###############
    #test_predictions = 1/((test_predictions+1)/test_data['pf_totalRaw'])#######################
#test_predictions = np.exp(test_predictions)
#test_predictions = test_predictions*test_max
    
#test_labels = 1/(np.exp(test_labels))*test_dataset['pf_totalRaw']#########################
    #test_labels = 1/((test_labels+1)/test_data['pf_totalRaw'])#################################
#test_labels = np.exp(test_labels)
#test_labels = test_labels*test_max

    return test_predictions, test_labels
    
