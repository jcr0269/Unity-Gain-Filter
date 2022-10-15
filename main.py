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

#Normalized signal
power = data/data.max()

#Ran for auto correlation
r = scipy.signal.correlate(power, power, 'full')

# playsound('gong.wav')
#Observation, the sound went gong.``
plt.figure(figsize=(15,8))
#data is for the origional sound wave
plt.plot(data)
plt.show()
plt.plot(r)
plt.show()
plt.psd(power)
plt.show()
