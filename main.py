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
data1=data
corr = np.correlate(data, data1, mode='valid')
Time = np.linspace(0, samplerate, len(corr))
# playsound('gong.wav')
#Observation, the sound went gong.
plt.figure(figsize=(15,8))
# plt.plot(data)
plt.plot(corr,Time)
plt.show()
