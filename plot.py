"""
2022/5/7
To plot the area rainfall of ledtime3 between 6.6 and 6.9 in 2020( histogram)
"""

import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

"""
（1）read obs
"""
file_obs = open("E:/2022/led3_out/obsn.txt", mode='r', encoding='utf-8')
line = file_obs.readline()
obs = []
while line:
    a = line.split()
    obs.append(a)
    line = file_obs.readline()
file_obs.close()

test_obs = obs[920:1196]  #test

year2010 = obs[920:920+92]
year2011 = obs[920+92:920+92*2]
year2012 = obs[920+92*2:920+92*3]


year2010 = np.array(year2010).astype(np.float)
obs2010  = np.mean(year2010, axis=1)
print(obs2010)

obs2010_06_06 = obs2010[5]
obs2010_06_07 = obs2010[6]
obs2010_06_08 = obs2010[7]
obs2010_06_09 = obs2010[8]
obs2010_06_10 = obs2010[9]


"""
read  rawmean
"""
def rawmean(path):
    file = open(path, mode='r', encoding='utf-8')
    line = file.readline()
    ledrawmean = []
    while line:
        a = line.split()
        ledrawmean.append(a)
        line = file.readline()
    file.close()

    data4 = np.array(ledrawmean, dtype=np.float64)
    data5 = data4.reshape(13*92, 225)
    rawmean_test = data5[92*10:92*13, :]
    rawmean = np.mean(rawmean_test, axis=1)
    rawmean_2010 = rawmean[0:92]
    return rawmean_2010

"""
read led3rawmean
"""
path1 = "E:/2022/led1-8_test/rawmean/"
filelist1 = os.listdir(path1)
le3_rawmean_2010_06_06 = rawmean(path1+"/"+filelist1[2])[5]
le3_rawmean_2010_06_07 = rawmean(path1+"/"+filelist1[2])[6]
le3_rawmean_2010_06_08 = rawmean(path1+"/"+filelist1[2])[7]
le3_rawmean_2010_06_09 = rawmean(path1+"/"+filelist1[2])[8]
le3_rawmean_2010_06_10 = rawmean(path1+"/"+filelist1[2])[9]


"""
read cnnmean、cnnmem、cnnspread
"""
def cnnmean(path):
    f = open(path, mode='r', encoding='utf-8')
    line = f.readline()
    data = []
    while line:
        a = line.split()
        data.append(a)
        line = f.readline()
    f.close()
    print(data)
    print("******************")
    data2 = np.array(data, dtype=np.float64)
    data2 = data2.reshape(92*3, 225)

    data3 = np.mean(data2, axis=1)
    cnnmean = list(data3)  #（276，）

    cnnmean_2010 = cnnmean[0:92]

    return cnnmean_2010


"""
cnnmean
"""
path2 = "E:/2022/led1-8_test/cnnmean_test/"
filelist2 = os.listdir(path2)

le3_cnnmean_2010_06_06 = cnnmean(path2+"/"+filelist2[2])[5]
le3_cnnmean_2010_06_07 = cnnmean(path2+"/"+filelist2[2])[6]
le3_cnnmean_2010_06_08 = cnnmean(path2+"/"+filelist2[2])[7]
le3_cnnmean_2010_06_09 = cnnmean(path2+"/"+filelist2[2])[8]
le3_cnnmean_2010_06_10 = cnnmean(path2+"/"+filelist2[2])[9]
"""
cnnmem
"""
path3 = "E:/2022/led1-8_test/cnnmem_test/"
filelist3 = os.listdir(path3)

le3_cnnmem_2010_06_06 = cnnmean(path3+"/"+filelist3[2])[5]
le3_cnnmem_2010_06_07 = cnnmean(path3+"/"+filelist3[2])[6]
le3_cnnmem_2010_06_08 = cnnmean(path3+"/"+filelist3[2])[7]
le3_cnnmem_2010_06_09 = cnnmean(path3+"/"+filelist3[2])[8]
le3_cnnmem_2010_06_10 = cnnmean(path3+"/"+filelist3[2])[9]
"""
cnnspread
"""
path4 = "E:/2022/led1-8_test/cnnspread_test/"
filelist4 = os.listdir(path4)
le3_cnnspread_2010_06_06 = cnnmean(path4+filelist4[2])[5]
le3_cnnspread_2010_06_07 = cnnmean(path4+filelist4[2])[6]
le3_cnnspread_2010_06_08 = cnnmean(path4+filelist4[2])[7]
le3_cnnspread_2010_06_09 = cnnmean(path4+filelist4[2])[8]
le3_cnnspread_2010_06_10 = cnnmean(path4+filelist4[2])[9]

"""
绘图
"""
le3_rawmean_data = [le3_rawmean_2010_06_07, le3_rawmean_2010_06_08,le3_rawmean_2010_06_09, le3_rawmean_2010_06_10]
le3_cnnmean_data = [le3_cnnmean_2010_06_07, le3_cnnmean_2010_06_08, le3_cnnmean_2010_06_09, le3_cnnmean_2010_06_10]
le3_cnnmem_data = [ le3_cnnmem_2010_06_07, le3_cnnmem_2010_06_08, le3_cnnmem_2010_06_09,le3_cnnmem_2010_06_10]
le3_cnnspread_data = [ le3_cnnspread_2010_06_07, le3_cnnspread_2010_06_08, le3_cnnspread_2010_06_09,le3_cnnspread_2010_06_10 ]
obsdata  =  [ obs2010_06_07, obs2010_06_08, obs2010_06_09,obs2010_06_10]


mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

x_data = ['06.07', '06.08', "06.09", "06.10"]
x = np.arange(len(x_data))  # the label locations
width = 0.15  # the width of the bars

fig = plt.figure(figsize=(7, 5))

plt.bar(x - 1.5*width, le3_rawmean_data, width, color="black", label='raw')
plt.bar(x - 0.5* width, le3_cnnmean_data, width, color="forestgreen", label='cnnmean')
plt.bar(x + 0.5*width, le3_cnnmem_data, width, color="brown",label='cnnmem')
plt.bar(x + 1.5*width, le3_cnnspread_data, width, color="orange", label='cnnspread')
plt.plot(x, obsdata, color='royalblue',  linewidth=2, linestyle='-', marker='o', markersize=8, label=u"obs")

plt.ylim(0, 45)
plt.ylabel('Mean Precipitation (mm/day)', fontsize=12)
# plt.title('leadtime=3 d', y=0.9,  fontsize=14)
plt.xticks([0, 1, 2, 3], [ '2010.06.07', '2010.06.08', "2010.06.09", "2010.06.10"])
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

fig.legend(loc="lower center", ncol=5,  fontsize=12, frameon=False)
plt.savefig('led3_2020_06_06_06_10.pdf')

plt.show()