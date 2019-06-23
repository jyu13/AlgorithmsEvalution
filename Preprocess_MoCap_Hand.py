# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 22:31:31 2019

@author:  Arnold Yu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv('data/allUsers.lcl.csv')

data = data.drop(['User'], axis = 1)
data = data.replace('?', 0).astype(np.float64)
# splict classes
data1 = data[data['Class'] == 1]
data2 = data[data['Class'] == 2]
data3 = data[data['Class'] == 3]
data4 = data[data['Class'] == 4]
data5 = data[data['Class'] == 5]

#data1['X11'].mean()
for col in data1:
    #print(col)
    data1[col] = data1[col].replace(0, data1[col].mean())
    data2[col] = data2[col].replace(0, data2[col].mean())
    data3[col] = data3[col].replace(0, data3[col].mean())
    data4[col] = data4[col].replace(0, data4[col].mean())
    data5[col] = data5[col].replace(0, data5[col].mean())


data_preprocess = pd.concat([data1,data2,data3,data4,data5])

data_preprocess.to_csv('data/allUser_preprocessed.csv', sep = ",", index = False)

# read = pd.read_csv('data/allUser_preprocessed.csv')

