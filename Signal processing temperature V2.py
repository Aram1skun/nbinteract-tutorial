# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 15:12:47 2018

@author: Cloutiern
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack 
from scipy.signal import butter, sosfiltfilt, sosfreqz

##########################################################
#Local variable definition
##########################################################

#Spindle rotation in Hertz 
#Note: Need an integer 
pf=21
#Sampling frequency in Hertz of the thermocouple acquisition
fs = 1000.0
#Order of the Butterworth bandpass filter
#Note:See graph to determine ne order to apply
order=15
#Highpass frequency of the filter
lowcut = 2
#Lowpass frequency of the filter
highcut = 50

#Selection of the data te be use in the analysis (application of a fixed window over the excell data)
#Note:Works independantly of the skiprows function in the reading of the excell
#Note:Indexing begin at 0
start=0
end=60000
N=end-start

########################################################
#Definition for the Butterworth bandpass filter
########################################################
def butter_bandpass(lowcut, highcut, fs, order=order):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
        return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=order):
        sos = butter_bandpass(lowcut, highcut, fs, order=order)
        y = sosfiltfilt(sos, data)
        return y

#######################################################
#Illustration of the filter frequency response
#######################################################
        
sos = butter_bandpass(lowcut, highcut, fs, order=order)
w, h = sosfreqz(sos, worN=2000)
plt.figure(1)
plt.clf()
for order in [3, 6, 9, 12, 15]:
        sos = butter_bandpass(lowcut, highcut, fs, order=order)
        w, h = sosfreqz(sos, worN=2000)
        plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.title('Butterworth filter frequency response')
plt.grid(True)
plt.legend(loc='best')
plt.axis([0,60,0,1.2])
plt.show()

#####################################################
#Extracting data from csv file
#Note:Column indexing start at 0
#Note:Skiprows eliminate the header in the excell
#####################################################

#Reading the filre with Pandas
df=pd.read_csv("/Users/Cloutiern/Desktop/f.csv",skiprows=8,usecols=[2])
#Removing axis label from the Pandas dataframe to retain only the values
data=df.values
#Shaping the dataframe to obtain an 1D array
data=data.ravel()
#Obtaining the size of the data import from the excell
I=data.size
#Keeping in memory the initial data import
datai=data

#####################################################
#Illustration of complete raw data from file
#####################################################

plt.figure(2)
xi = np.linspace(0.0,I/fs,I)
plt.plot(xi,data)
plt.title('Complete raw data')
plt.ylabel('Volt')
plt.xlabel('Time')
plt.show()

#####################################################
#Numpy array slicing numpyarray[start:end:step]
#####################################################
#Note:Numpy array indexing begin at 0
data=data[start:end:1]

#####################################################
#Applying butterworth bandpass filter to data
#####################################################
dataf = butter_bandpass_filter(data,lowcut, highcut, fs, order=order)

#####################################################
#Some graph
#####################################################

#illustration of figure 3
plt.figure(3)
x = np.linspace(0.0, N/fs, N)

#Raw data graph
plt.subplot(321)
plt.plot(x,data)
plt.title('Raw data')
plt.ylabel('Volt')
plt.xlabel('Time')

#Filtered data
plt.subplot(322)
plt.plot(x,dataf)
plt.title('Filtered data')
plt.ylabel('Volt')
plt.xlabel('Time')

######################################################
#Fourrier transform FFT
######################################################

dataid=scipy.signal.detrend(datai)
yi = scipy.fftpack.fft(datai)
yf = scipy.fftpack.fft(data)
yff= scipy.fftpack.fft(dataf)
yidf=scipy.fftpack.fft(dataid)
xf = np.linspace(0.0, 1.0/(2.0/fs), N/2)
xif = np.linspace(0.0, 1.0/(2.0/fs), I/2)

plt.subplot(312)
plt.plot(xf,2.0/N * np.abs(yff[:N//2]))
plt.title('FFT filtered')
plt.ylabel('Density')
plt.xlabel('Frequency')
plt.yscale('linear')
plt.xscale('linear')
plt.axis([0,40,0,0.003])

plt.subplot(313)
plt.plot(xf,2.0/N * np.abs(yf[:N//2]))
plt.title('FFT')
plt.ylabel('Density')
plt.xlabel('Frequency')
plt.yscale('linear')
plt.xscale('linear')
plt.axis([0,150,0,0.1])
plt.show()

plt.figure(4)

plt.subplot(311)
xi = np.linspace(0.0,I/fs,I)
plt.plot(xi,datai)

plt.subplot(312)
plt.plot(xf,2.0/N * np.abs(yf[:N//2]))
plt.xscale('log')
plt.axis([0,fs/2,0,0.005])
plt.title('FFT unfiltered log x axis')

plt.subplot(313)
plt.plot(xif,2.0/I * np.abs(yi[:I//2]))
plt.title('FFT')
plt.ylabel('Density')
plt.xlabel('Frequency')
plt.yscale('linear')
plt.xscale('linear')
plt.axis([0,(1*fs)/2,0,0.003])
plt.show()

plt.figure(5)
plt.plot(xif,2.0/I * np.abs(yidf[:I//2]))
plt.title('FFT')
plt.ylabel('Density')
plt.xlabel('Frequency')
plt.yscale('linear')
plt.xscale('linear')
plt.axis([0,(1*fs)/2,0,0.003])
plt.show()

plt.figure(6)
plt.plot(xif,2.0/I * np.abs(yidf[:I//2]))
plt.title('FFT')
plt.ylabel('Density')
plt.xlabel('Frequency')
plt.yscale('linear')
plt.xscale('linear')
plt.axis([0,30,0,0.003])
plt.show()

###################################################
#Analysis of the temperature delta across temperature profil
###################################################

#Creation of a numpy array containg the filtered data
d=dataf
#Reshaping the array to have row of lenght of the spindle frequency in Hertz
d=np.reshape(d,(int(N/pf),pf))
#Function calculating the maximal difference on a row (a complete tool rotation)
def max_minus_min(x):
    return np.max(x)-np.min(x)
#Application of the delta dunction on each row
d=np.apply_along_axis(max_minus_min,1,d)

#Detrending if the data shows a DC composant
#Note:to replace a highpass filter non-applicable in this case
#d=scipy.signal.detrend(d)
#If there is negative temperature difference
#d=np.absolute(d)

#Illustration of the temperature delta for each rotation of the tool
xd = np.arange(N/pf)
plt.plot(xd,d)
plt.title('Temperature delta per cycle')
plt.ylabel('Temperature delta')
plt.xlabel('Cycle')
#plt.axis([0,N/pf,0,0.005])
plt.show()

#Illustration of the FFT of the temperature delta for each rotation of the tool
xdfft = np.linspace(0.0, 1.0/(2.0/pf), (N/pf)/2)
yd = scipy.fftpack.fft(d)
plt.plot(xdfft,2.0/(N/pf) * np.abs(yd[:int(N/pf)//2]))
plt.title('FFT of temperature difference per cycle')
plt.ylabel('Density')
plt.xlabel('Frequency')
plt.yscale('linear')
plt.xscale('linear')
#plt.axis([0,0,0,0])
plt.show()

