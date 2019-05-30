# coding: utf-8
#计算主题在各时间区间的支持文档数，以民生新闻为例，输入为HDP主题建模的到的theta矩阵，输出为支持文档数矩阵

import csv
import numpy as np
import math
from numpy.random import choice
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
get_ipython().magic('matplotlib inline')

def newcsv(infile,outfile):
    f1 = pd.read_csv(infile)
    f2 = f1.drop(['Unnamed: 0'],axis=1)
    f2.to_csv(outfile,index=False,sep=',')
    return f2

infile = "./d_data/1.pepnews/theta13t16.csv"
outfile = "./d_data/1.pepnews/new-theta113t16.csv"
ff = newcsv(infile,outfile)
f2 = pd.read_csv(outfile)

#按季度划分，分为16季度，2013-2016年,民生新闻
data1 = f2.loc[0:2024]
data2 = f2.loc[2025:3744]
data3 = f2.loc[3745:5378]
data4 = f2.loc[5379:7538]
data5 = f2.loc[7539:10355]
data6 = f2.loc[10356:11687]
data7 = f2.loc[11688:13031]
data8 = f2.loc[13032:14073]
data9 = f2.loc[14074:14593]
data10 = f2.loc[14594:14911]
data11 = f2.loc[14912:15296]
data12 = f2.loc[15297:15750]
data13 = f2.loc[15751:16165]
data14 = f2.loc[16166:16761]
data15 = f2.loc[16762:17562]
data16 = f2.loc[17563:18374]
datalist = [data1,data2,data3,data4,data5,data6,data7,data8,data9,data10,data11,data12,data13,data14,data15,data16]

def fu(datalist,threshold):
    a = np.zeros((16,50))
    for k in range(50):
        for i,data in enumerate(datalist):
            tvec = data
            new = tvec.loc[tvec[str(k)]>threshold]
            l = len(new)
            a[i][k] = l
    return a

a1 = fu(datalist,0.1)
outfile1 =  "./d_data/1.pepnews/emg_topic/pep_RVI.csv"
np.savetxt(outfile1, a1, delimiter = ',')
