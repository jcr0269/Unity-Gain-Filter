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
f1 = 30  # Frequency of 1st signal in Hz
f2 = 30
#Normalized signal
power = data/data.max()
n = np.linspace(0, 1, 1000)
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

#filter


#I made a filter for 60Hz and it hurt
samp_freq = 1000  # Sample frequency (Hz)
notch_freq = 50.0  # Frequency to be removed from signal (Hz)
quality_factor = 20.0  # Quality factor
b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, samp_freq)

# Compute magnitude response of the designed filter
freq, h = signal.freqz(b_notch, a_notch, fs=2 * np.pi)
x = (freq*samp_freq/(2*np.pi), 20 * np.log10(abs(h)))

outputSignal = signal.filtfilt(b_notch, a_notch, new_data)
# h = scipy.signal.firwin(121, 60, nyq=length/2)
# h = np.convolve(r_new_data,h)
length1 = np.linspace(0,samplerate, len(outputSignal))
#data is for the origional sound wave
# plt.plot(data)
# plt.show()
# plt.plot(r)
# plt.show()
# #psd is welch
# plt.psd(power)
# plt.show()
# plt.plot(new_data)
# plt.show()
# plt.plot(r_new_data)
# plt.show()
# plt.psd(r_new_data)
# plt.show()
#the filter part
plt.plot(freq*samp_freq/(2*np.pi), 20 * np.log10(abs(h)),
         'r', label='Bandpass filter', linewidth='2')
plt.show()

plt.plot(length1, outputSignal)
plt.show()
