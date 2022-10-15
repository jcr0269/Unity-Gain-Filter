import scipy.signal
from scipy.io import wavfile
from playsound import playsound
import soundfile as sf
import scipy.signal as signal
import numpy as np
import matplotlib
# matplotlib.use('TkAgg', force=True)

import matplotlib.pyplot as plt
samplerate, data = wavfile.read('chirp.wav')

#Normalized signal
power = data/data.max()

#Ran for auto correlation
r = scipy.signal.correlate(power, power, 'full')

#making a sine wave and adding it to data
data_a = np.array(data)
length = len(data)
t = np.linspace(0, 1000, num=length)
the_wave = .5*np.sin(2*np.pi*60*t)
new_data = the_wave + data_a

new_data_norm = new_data/new_data.max()
r_new_data = scipy.signal.correlate(new_data_norm,new_data_norm,'full')
# playsound('gong.wav')
#Observation, the sound went gong.``

#data is for the origional sound wave
plt.plot(data)
plt.show()
plt.plot(r)
plt.show()
#psd is welch
plt.psd(power)
plt.show()
plt.plot(new_data)
plt.show()
plt.plot(r_new_data)
plt.show()
