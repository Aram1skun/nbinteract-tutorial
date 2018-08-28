# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

#Selection of the data te be use in the analysis (application of a fixed window over the excell data)
#Note:Works independantly of the skiprows function in the reading of the excell
#Note:Indexing begin at 0
start=0
end=1000
N=end-start

#####################################################
#Extracting data from csv file
#Note:Column indexing start at 0
#Note:Skiprows eliminate the header in the excell
#####################################################

#Reading the filre with Pandas
dfx=pd.read_csv("/Users/Cloutiern/Desktop/p2.csv",skiprows=8,usecols=[1])
#Removing axis label from the Pandas dataframe to retain only the values
datax=dfx.values
#Shaping the dataframe to obtain an 1D array
datax=datax.ravel()

#Reading the filre with Pandas
dfy=pd.read_csv("/Users/Cloutiern/Desktop/p2.csv",skiprows=8,usecols=[2])
#Removing axis label from the Pandas dataframe to retain only the values
datay=dfy.values
#Shaping the dataframe to obtain an 1D array
datay=datay.ravel()


#Reading the filre with Pandas
dfz=pd.read_csv("/Users/Cloutiern/Desktop/p2.csv",skiprows=8,usecols=[3])
#Removing axis label from the Pandas dataframe to retain only the values
dataz=dfz.values
#Shaping the dataframe to obtain an 1D array
dataz=dataz.ravel()


#Reading the filre with Pandas
dfw=pd.read_csv("/Users/Cloutiern/Desktop/p2.csv",skiprows=8,usecols=[9])
#Removing axis label from the Pandas dataframe to retain only the values
dataw=dfw.values
#Shaping the dataframe to obtain an 1D array
dataw=dataw.ravel()

print(datax)
print(datay)
print(dataz)
print(dataw)

#Numpy array slicing numpyarray[start:end:step]
#####################################################
#Note:Numpy array indexing begin at 0
datax=datax[start:end:1]
datay=datay[start:end:1]
dataz=dataz[start:end:1]
dataw=dataw[start:end:1]


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = datax
y = datay
z = dataz
c = dataw

d = cm.ScalarMappable(cmap=cm.Oranges)
d.set_array(dataw)
cbar = plt.colorbar(d)

ax.scatter(x, y, z, c=c, cmap=plt.hot())

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()

