import scipy as scipy
import scipy.signal
from scipy.io import wavfile
import soundfile as sf
import scipy.signal as signal
import numpy as np
# import matplotlib
# matplotlib.use('TkAgg', force=True)
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.figure import Figure
from matplotlib import rcParams

samplerate, data = wavfile.read('chirp.wav')
f1 = 30  # Frequency of 1st signal in Hz
f2 = 30
#Normalized signal
power = 1 - data.astype(float)/2**7
n = np.linspace(0, 1, 8192)
#Ran for auto correlation
first_corr = scipy.signal.correlate(power, power, 'full')

# sd.play(data, samplerate=samplerate)

#making a sine wave and adding it to data

data_a = data/data.max()
data_a = 1 - data.astype(float)/2**7
length = len(data)
t = np.linspace(0, 1/samplerate, num=length)
the_wave = .5*(np.sin(2*np.pi*60*t))

new_data = the_wave + data_a

# sd.play(new_data, samplerate=samplerate)

r_new_data = scipy.signal.correlate(new_data,new_data,'full')

#I made a filter for 60Hz and it hurt
samp_freq = 1000  # Sample frequency (Hz)
notch_freq = (60.0/samplerate/2)  # Frequency to be removed from signal (Hz)
quality_factor = 1.0  # Quality factor
#b_notch is numerator
b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor)

# Compute magnitude response of the designed filter
freq, h = signal.freqz(b_notch, a_notch, fs=2 * np.pi)
x = (freq*samp_freq/(2*np.pi), 20 * np.log10(abs(h)))

outputSignal = signal.filtfilt(b_notch, a_notch, new_data)
length_out = len(outputSignal)
length1 = np.linspace(0, 1/samplerate, length_out)

######################

# Copyright (c) 2011 Christopher Felton
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

# The following is derived from the slides presented by
# Alexander Kain for CS506/606 "Special Topics: Speech Signal Processing"
# CSLU / OHSU, Spring Term 2011.
a = a_notch
b = b_notch

def zplane(b, a, filename=None):
    """Plot the complex z-plane given a transfer function.
    """

    # get a figure/plot
    ax = plt.subplot(111)

    # create the unit circle
    uc = patches.Circle((0, 0), radius=1, fill=False,
                        color='black', ls='dashed')
    ax.add_patch(uc)

    # The coefficients are less than 1, normalize the coeficients
    if np.max(b) > 1:
        kn = np.max(b)
        b = b / float(kn)
    else:
        kn = 1

    if np.max(a) > 1:
        kd = np.max(a)
        a = a / float(kd)
    else:
        kd = 1

    # Get the poles and zeros
    p = np.roots(a)
    z = np.roots(b)
    k = kn / float(kd)

    # Plot the zeros and set marker properties
    t1 = plt.plot(z.real, z.imag, 'go', ms=10)
    plt.setp(t1, markersize=10.0, markeredgewidth=1.0,
             markeredgecolor='k', markerfacecolor='g')

    # Plot the poles and set marker properties
    t2 = plt.plot(p.real, p.imag, 'rx', ms=10)
    plt.setp(t2, markersize=12.0, markeredgewidth=3.0,
             markeredgecolor='r', markerfacecolor='r')
    print(z, p)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # set the ticks
    r = 1.5;
    plt.axis('scaled');
    plt.axis([-r, r, -r, r])
    ticks = [-1, -.5, .5, 1];
    plt.xticks(ticks);
    plt.yticks(ticks)

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)

    return z, p, k


################NUMBER 5##############################
# num5 = np.convolve(outputSignal, new_data, mode='full')
num5 = signal.filtfilt(b_notch, a_notch, new_data)
# num5_1 = num5/num5.max()
num5_1 = np.linalg.norm(num5)
num5_1 = num5/num5_1
num5_1 = signal.correlate(num5_1, num5_1)

#the length
T_5 = np.linspace(0, 1, len(num5_1))

##################NUMBER 6############################
#https://www.youtube.com/watch?v=ke4i9NsTOyQ
fs = samplerate
r = 0.95
ct_notch_f = 60
dt_notch_f = 2*np.pi*ct_notch_f/fs
notch_zeros = [np.exp(1j*dt_notch_f), np.exp(-1j*dt_notch_f)]
notch_poles = [r*np.exp(1j*dt_notch_f), r*np.exp(-1j*dt_notch_f)]
denom = np.poly(notch_zeros)
numer = np.poly(notch_poles)
width, height = scipy.signal.freqz(denom, numer, fs=fs)
#
zplane(denom, numer)
zplane(b,a)

#############NUMBER 7##################################
num8 = signal.filtfilt(denom, numer, new_data)
# sd.play(num8, samplerate=samplerate)
num8_1 = 1 - num8.astype(float)/2**7
num8_1 = signal.correlate(num8_1, num8_1)

# data is for the origional sound wave
plt.plot(data)
plt.xlabel("Original Sound Wave")
plt.show()
plt.plot(first_corr)
plt.xlabel("Auto Correlation of original")
plt.savefig('auto of orignial')
plt.show()
#psd is welch
plt.psd(power)
plt.xlabel("this is the PSD of original")
plt.savefig('psd of original ')
plt.show()
plt.plot(new_data)
plt.xlabel("orignal + 60Hz Sine wave")
plt.savefig('original + sine')
plt.show()
plt.plot(r_new_data)
plt.xlabel("Auto Correlation of Corrupted Signal")
plt.savefig('Auto Correlation of corrupted')
plt.show()
plt.psd(r_new_data)
plt.xlabel("psd of corrupted")
plt.savefig('psd of corrupted')
plt.show()
#the filter part
plt.plot(freq, 20 * np.log10(abs(h)),
         'r', label='Bandpass filter', linewidth='2')
plt.savefig('bandpass')
plt.show()
plt.plot(length1, outputSignal)
plt.xlabel("Got signal back")
plt.savefig('Got Signal Back')
plt.show()
plt.plot(num5)
plt.xlabel("corrupted with notch")
plt.savefig('corrupted with notch')
plt.show()
plt.plot(T_5, num5_1)
plt.xlabel("corrupted with notch")
plt.savefig('corrupted with notch')
plt.show()
plt.psd(num5_1)
plt.xlabel("Power Density of Number 5")
plt.savefig('Power Density of Number 5')
plt.show()
plt.plot(width/np.pi, height)
plt.xlabel("Number 6a")
plt.savefig('number 6a')
plt.show()
plt.plot(num8)
plt.xlabel('number 7 a')
plt.savefig('number 7a')
plt.show()
plt.plot(num8_1)
plt.xlabel('number 7 b')
plt.savefig('number 7b')
plt.show()

plt.psd(num8_1)
plt.xlabel('number 7 c')
plt.savefig('number 7c')
plt.show()




