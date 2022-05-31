import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
#########readtxt########

file_obs = open("/home/linuxrs/algorithm/testyang/data/led1-8memcnn.txt", mode='r', encoding='utf-8')
line = file_obs.readline()
data = []
while line:
    a = line.split()
    data.append(a)
    line = file_obs.readline()
file_obs.close()
#print(data)
cnnl = np.array(data, dtype=float)  
cnnl= cnnl.reshape(8, 1196, 225)

led1cnn=cnnl[0,920:1196,:]
led1cnn=led1cnn.reshape(276*225,1)
led2cnn=cnnl[1,920:1196,:]
led2cnn=led2cnn.reshape(276*225,1)
led3cnn=cnnl[2,920:1196,:]
led3cnn=led3cnn.reshape(276*225,1)
led4cnn=cnnl[3,920:1196,:]
led4cnn=led4cnn.reshape(276*225,1)
led5cnn=cnnl[4,920:1196,:]
led5cnn=led5cnn.reshape(276*225,1)
led6cnn=cnnl[5,920:1196,:]
led6cnn=led6cnn.reshape(276*225,1)
led7cnn=cnnl[6,920:1196,:]
led7cnn=led7cnn.reshape(276*225,1)
led8cnn=cnnl[7,920:1196,:]
led8cnn=led8cnn.reshape(276*225,1)

file_obs = open("/home/linuxrs/algorithm/testyang/data/led1-8raw.txt", mode='r', encoding='utf-8')
line = file_obs.readline()
data1 = []
while line:
    a = line.split()
    data1.append(a)
    line = file_obs.readline()
file_obs.close()
#print(data)
raw = np.array(data1, dtype=float) 
raw= raw.reshape(8, 1196, 225)

led1raw=raw[0,920:1196,:]
led1raw=led1raw.reshape(276*225,1)
led2raw=raw[1,920:1196,:]
led2raw=led2raw.reshape(276*225,1)
led3raw=raw[2,920:1196,:]
led3raw=led3raw.reshape(276*225,1)
led4raw=raw[3,920:1196,:]
led4raw=led4raw.reshape(276*225,1)
led5raw=raw[4,920:1196,:]
led5raw=led5raw.reshape(276*225,1)
led6raw=raw[5,920:1196,:]
led6raw=led6raw.reshape(276*225,1)
led7raw=raw[6,920:1196,:]
led7raw=led7raw.reshape(276*225,1)
led8raw=raw[7,920:1196,:]
led8raw=led8raw.reshape(276*225,1)
###########################################################

file_obs = open("/home/linuxrs/algorithm/testyang/data/obsn.txt", mode='r', encoding='utf-8')
line = file_obs.readline()
obs = []
while line:
    a = line.split()
    obs.append(a)
    line = file_obs.readline()
file_obs.close()
test_obs = obs[920:1196]  

test_obs = np.array(test_obs)
test_obs = test_obs.astype(np.float)
test_obs=test_obs.reshape(276*225,1)
obsn=test_obs
#########
def prep_clf(obs,pre, threshold):
    '''
    func: 
    inputs:
        obs: 
        pre: 
        threshold: 
    
    returns:
        hits, misses, falsealarms, correctnegatives
        #aliases: TP, FN, FP, TN 
    '''

    obs = np.where(obs >= threshold, 1, 0)
    pre = np.where(pre >= threshold, 1, 0)

    # True positive (TP)
    hits = np.sum((obs == 1) & (pre == 1))

    # False negative (FN)
    misses = np.sum((obs == 1) & (pre == 0))

    # False positive (FP)
    falsealarms = np.sum((obs == 0) & (pre == 1))

    # True negative (TN)
    correctnegatives = np.sum((obs == 0) & (pre == 0))

    return hits, misses, falsealarms, correctnegatives


def precision(obs, pre, threshold):
    '''
    func:  TP / (TP + FP)
    inputs:
        obs:
        pre: 
        threshold: 
    
    returns:
        dtype: float
    '''

    TP, FN, FP, TN = prep_clf(obs=obs, pre = pre, threshold=threshold)

    return TP / (TP + FP)


def recall(obs, pre, threshold):
    '''
    func: TP / (TP + FN)
    inputs:
        obs: 
        pre: 
        threshold: 
    
    returns:
        dtype: float
    '''

    TP, FN, FP, TN = prep_clf(obs=obs, pre = pre, threshold=threshold)

    return TP / (TP + FN)


def ACC(obs, pre, threshold):
    '''
    func: Accuracy: (TP + TN) / (TP + TN + FP + FN)
    inputs:
        obs: 
        pre: 
        threshold: 
    
    returns:
        dtype: float
    '''

    TP, FN, FP, TN = prep_clf(obs=obs, pre = pre, threshold=threshold)

    return (TP + TN) / (TP + TN + FP + FN)

def FSC(obs, pre, threshold):
    '''
    func:计算f1 score = 2 * ((precision * recall) / (precision + recall))
    '''
    precision_socre = precision(obs, pre, threshold=threshold)
    recall_score = recall(obs, pre, threshold=threshold)

    return 2 * ((precision_socre * recall_score) / (precision_socre + recall_score))

def ETS(obs, pre, threshold):
    '''
    ETS - Equitable Threat Score
    details in the paper:
    Winterrath, T., & Rosenow, W. (2007). A new module for the tracking of
    radar-derived precipitation with model-derived winds.
    Advances in Geosciences,10, 77–83. https://doi.org/10.5194/adgeo-10-77-2007
    Args:
        obs (numpy.ndarray): observations
        pre (numpy.ndarray): prediction
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)
    Returns:
        float: ETS value
    '''
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre = pre,
                                                           threshold=threshold)
    num = (hits + falsealarms) * (hits + misses)
    den = hits + misses + falsealarms + correctnegatives
    Dr = num / den

    ETS = (hits - Dr) / (hits + misses + falsealarms - Dr)

    return ETS
def FAR(obs, pre, threshold):
    '''
    func: falsealarms / (hits + falsealarms) 
    FAR - false alarm rate
    Args:
        obs (numpy.ndarray): observations
        pre (numpy.ndarray): prediction
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)
    Returns:
        float: FAR value
    '''
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre = pre,
                                                           threshold=threshold)

    return falsealarms / (hits + falsealarms)
def MAR(obs, pre, threshold):
    '''
    func : misses / (hits + misses)
    MAR - Missing Alarm Rate
    Args:
        obs (numpy.ndarray): observations
        pre (numpy.ndarray): prediction
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)
    Returns:
        float: MAR value
    '''
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre = pre,
                                                           threshold=threshold)

    return misses / (hits + misses)
def POD(obs, pre, threshold):
    '''
    func :hits / (hits + misses)
    pod - Probability of Detection
    Args:
        obs (numpy.ndarray): observations
        pre (numpy.ndarray): prediction
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)
    Returns:
        float: PDO value
    '''
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre = pre,
                                                           threshold=threshold)

    return hits / (hits + misses)

def BIAS(obs, pre, threshold):
    '''
    func:Bias =  (hits + falsealarms)/(hits + misses) 
    	  alias: (TP + FP)/(TP + FN)
    inputs:
        obs: 
        pre: 
        threshold: 
    returns:
        dtype: float
    '''    
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre = pre,
                                                           threshold=threshold)

    return (hits + falsealarms) / (hits + misses)
def BSS(obs, pre, threshold):
    '''
    BSS - Brier skill score
    Args:
        obs (numpy.ndarray): observations
        pre (numpy.ndarray): prediction
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)
    Returns:
        float: BSS value
    '''
    obs = np.where(obs >= threshold, 1, 0)
    pre = np.where(pre >= threshold, 1, 0)

    obs = obs.flatten()
    pre = pre.flatten()

    return np.sqrt(np.mean((obs - pre) ** 2))
#####################
etsraw = np.zeros((4, 8))#######LED1-8
etsraw[0,0]=BIAS(obsn,led1raw,0.1)
etsraw[1,0]=BIAS(obsn,led1raw,10)
etsraw[2,0]=BIAS(obsn,led1raw,25)
etsraw[3,0]=BIAS(obsn,led1raw,50)

etsraw[0,1]=BIAS(obsn,led2raw,0.1)
etsraw[1,1]=BIAS(obsn,led2raw,10)
etsraw[2,1]=BIAS(obsn,led2raw,25)
etsraw[3,1]=BIAS(obsn,led2raw,50)

etsraw[0,2]=BIAS(obsn,led3raw,0.1)
etsraw[1,2]=BIAS(obsn,led3raw,10)
etsraw[2,2]=BIAS(obsn,led3raw,25)
etsraw[3,2]=BIAS(obsn,led3raw,50)

etsraw[0,3]=BIAS(obsn,led4raw,0.1)
etsraw[1,3]=BIAS(obsn,led4raw,10)
etsraw[2,3]=BIAS(obsn,led4raw,25)
etsraw[3,3]=BIAS(obsn,led4raw,50)

etsraw[0,4]=BIAS(obsn,led5raw,0.1)
etsraw[1,4]=BIAS(obsn,led5raw,10)
etsraw[2,4]=BIAS(obsn,led5raw,25)
etsraw[3,4]=BIAS(obsn,led5raw,50)

etsraw[0,5]=BIAS(obsn,led6raw,0.1)
etsraw[1,5]=BIAS(obsn,led6raw,10)
etsraw[2,5]=BIAS(obsn,led6raw,25)
etsraw[3,5]=BIAS(obsn,led6raw,50)

etsraw[0,6]=BIAS(obsn,led7raw,0.1)
etsraw[1,6]=BIAS(obsn,led7raw,10)
etsraw[2,6]=BIAS(obsn,led7raw,25)
etsraw[3,6]=BIAS(obsn,led7raw,50)

etsraw[0,7]=BIAS(obsn,led8raw,0.1)
etsraw[1,7]=BIAS(obsn,led8raw,10)
etsraw[2,7]=BIAS(obsn,led8raw,25)
etsraw[3,7]=BIAS(obsn,led8raw,50)


np.savetxt('biasraw.txt',etsraw)
