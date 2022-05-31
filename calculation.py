"""
2021/4/15
Calculate RMSE, CC and re of predicted precipitation and real precipitation
"""
import itertools
import os
import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)
os.chdir("E:/14DL-unet-Regression/run/return/")

#####################################Read the predicted quantified precipitation file############################################

f = open("./led7rawmean_2/b12_e150_final_test_pre.txt", mode='r', encoding='utf-8')
line = f.readline()
print(line)
print(type(line)) #str
line = line.split()
list = list(line)

pre = []
for i in range(len(list)):
    pre.append(float(list[i]))
print(pre)
pre = np.array(pre)
pre = pre.astype(float)

###################################################Read forecast data################################################################3

f = open("E:/14DL-unet-Regression/data_make/led7rawmean.txt", mode='r', encoding='utf-8')
line = f.readline()
print(line)
print(type(line))
led1 = line.split()
led1_data = []
for i in range(len(led1)):
    led1_data.append(float(led1[i]))

led1_data = np.array(led1_data)
led1_data = led1_data.reshape(1196, 225)



led1_data = led1_data[920:1196, :] #test
#led1_data = np.divide(led1_data, 500.0)

#led1_data = led1_data[0:736, :] #train

#led1_data = led1_data[736:920, :]   #val

led1_data =led1_data.reshape(-1)
led1_data = led1_data.astype(float)

##############################################Read real observed precipitation data（obsn.txt）###########################################################

file_obs = open("E:/14DL-unet-Regression/data_make/obsn.txt", mode='r', encoding='utf-8')
line = file_obs.readline()
obs = []
while line:
    a = line.split()
    obs.append(a)
    line = file_obs.readline()
file_obs.close()
print(obs)

print(len(obs[920:1196]))
test_obs = obs[920:1196]  #test

#test_obs = np.divide(led1_data, 500.0)
#test_obs = obs[0:736] #train
#test_obs = obs[736:920] #val
test_obs = np.array(test_obs)
test_obs = test_obs.reshape(-1)
test_obs = test_obs.astype(np.float)
# Calculate  rmse、cc、re
#The input is arrays, and the type must be unified (two-dimensional array becomes one-dimensional array)

def mse(y_true, y_pred):
    m = np.mean((y_true-y_pred)**2)
    return m
def rmse(y_true,y_pred):
    m = mse(y_true, y_pred)
    if mse:
        return math.sqrt(m)
    else:
        return None
def cc(y_true, y_pred):
    y_true_mean = np.mean(y_true)
    y_pred_mean = np.mean(y_pred)
    s = np.sum((y_true-y_true_mean)*(y_pred-y_pred_mean))
    t = np.sqrt(np.sum((y_true-y_true_mean)**2))
    p = np.sqrt(np.sum((y_pred-y_pred_mean)**2))
    c = s/(t*p)
    return c
def bais(y_true, y_pred):
    bais = np.mean(y_pred-y_true)
    return bais

def re(y_true, y_pred):
    r = np.sum(y_pred-y_true)
    o = np.sum(y_true)
    re = (r/o)*100
    return re

print("__"*20)
print("Deep learning predicted precipitation data and real observed precipitation data")
print("rmse：", rmse(test_obs, pre))
print("cc：", cc(test_obs, pre))
print("bais：", bais(test_obs, pre))
print("re：", re(test_obs, pre))

print("__"*20)
print("Forecast precipitation data and real observed precipitation data")
print("rmse：", rmse(test_obs, led1_data))
print("cc：", cc(test_obs, led1_data))
print("bais：", bais(test_obs, led1_data))
print("re：", re(test_obs, led1_data))
print("__"*20)




