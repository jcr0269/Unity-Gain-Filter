import scipy.signal
from scipy.io import wavfile
from playsound import playsound
import soundfile as sf
import scipy.signal as signal
import numpy as np
import matplotlib
# matplotlib.use('TkAgg', force=True)
import matplotlib.pyplot as plt
samplerate, data = wavfile.read('gong.wav')

#autocorrelation of the .wav file
# data_a = np.float32(data)
# corr = signal.correlate(data_a, data_a)
# lags = signal.correlation_lags(len(data_a), len(data_a))
# corr = corr / np.max(corr)
# lag = lags[np.argmax(corr)]
################

#Corralate take 2 - This sort of works but need to dig into more.
# r = np.correlate(data, data, mode='full')[len(data)-1:]

#Ran for auto correlation
r = scipy.signal.correlate(data, data, 'full')

# data1=data
# corr = np.correlate(data, data1, mode='valid')
# Time = np.linspace(0, samplerate, len(corr))
# playsound('gong.wav')

#Observation, the sound went gong.
plt.figure(figsize=(15,8))
#data is for the origional sound wave
# plt.plot(data)
#x limit so that the wave form isnt to big on the plot
plt.xlim([40000,44000])
plt.plot(r)
plt.show()
