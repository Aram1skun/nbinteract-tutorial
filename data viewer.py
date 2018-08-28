# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 14:54:42 2018

@author: cloutiern
"""

import plotly
print(plotly.__version__)

import plotly
import plotly.graph_objs as go
import pandas as pd

#Reading the filre with Pandas
dfx=pd.read_csv("/Users/Cloutiern/Desktop/c.csv",skiprows=8,usecols=[0])
#Removing axis label from the Pandas dataframe to retain only the values
datax=dfx.values
#Shaping the dataframe to obtain an 1D array
datax=datax.ravel()

#Reading the filre with Pandas
dfy=pd.read_csv("/Users/Cloutiern/Desktop/c.csv",skiprows=8,usecols=[2])
#Removing axis label from the Pandas dataframe to retain only the values
datay=dfy.values
#Shaping the dataframe to obtain an 1D array
datay=datay.ravel()


trace = go.Scatter(x=datax, y=datay)

data = [trace]
layout = dict(
    title='Time series with range slider and selectors',
    xaxis=dict(
        rangeselector=dict(
#            buttons=list([
#                dict(count=1,
#                     label='1',
#                     step='minute',
#                     stepmode='backward'),
#                dict(count=10,
#                     label='10',
#                     step='minute',
#                     stepmode='backward'),
#                dict(count=100,
#                    label='100',
#                    step='minute'),
#                dict(count=1000,
#                    label='1k',
#                    step='minute',
#                    stepmode='backward'),
#                dict(step='all')
#                       ])
        ),
        rangeslider=dict(
            visible = True
        ),
#        type='date'
    )
)

plotly.offline.plot({
        

    "data": data,
    "layout": layout
    
}, auto_open=True)

import plotly.plotly as py
import plotly.graph_objs as go 








fig = dict(data=data, layout=layout)
py.iplot(fig)