import os
from os.path import isdir, join
from pathlib import Path
import pandas as pds

# Math
import numpy as np

from scipy.fftpack import fft
from scipy import signal
from scipy.io import wavfile
import librosa

from sklearn.decomposition import PCA

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import IPython.display as ipd
import librosa.display

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import pandas as pd


class SPEACH_PRO(object):

    def __init__(self):
        self.train_audio_path = './data/'
        self.filename = '/001.wav'
        self.samples= None
        self.sample_rate =None
        self.freqs = None
        self.times = None
        self.spectrogram = None
        self.filename_list = []
        self.spectrogram_list = []

    def log_specgram(self, audio, sample_rate, window_size=20,
                     step_size=10, eps=1e-10):
        nperseg = int(round(window_size * sample_rate / 1e3))
        noverlap = int(round(step_size * sample_rate / 1e3))
        freqs, times, spec = signal.spectrogram(audio,
                                                fs=sample_rate,
                                                window='hann',
                                                nperseg=nperseg,
                                                noverlap=noverlap,
                                                detrend=False)
        return freqs, times, np.log(spec.T.astype(np.float32) + eps)

    def set_sample(self,duration):
        for i in range(len(self.filename_list)):
            self.samples, self.sample_rate = librosa.load(str(self.train_audio_path) + self.filename_list[0][i], duration=duration)
            self.freqs, self.times, self.spectrogram = self.log_specgram(self.samples, self.sample_rate)
            self.spectrogram_list.append(self.spectrogram)

    def re_sampling(self, target_shape=None):
        if not target_shape:
            target_shape = self.sample_rate
        sample_shape = self.samples.shape[0]
        assert target_shape < sample_shape
        step_size = sample_shape // target_shape
        re_samples = [self.samples[i] for i in range(0, sample_shape, step_size)]
        re_samples = re_samples[:target_shape]
        return np.array(re_samples)

    def show_plot(self):
        plt.figure()
        plt.imshow(self.spectrogram.T, aspect='auto', origin='lower',
                   extent=[self.times.min(), self.times.max(), self.freqs.min(), self.freqs.max()])
        plt.savefig('result.png')
        plt.show()

    def init_data(self):
        if os.path.isdir('./data'):
            waves = [f for f in os.listdir(join(self.train_audio_path, '')) if f.endswith('.wav')]
            self.filename_list.append(waves)




if __name__=="__main__":
    model = SPEACH_PRO()
    model.init_data()
    model.set_sample(20)
    print(model.spectrogram_list)
    print(np.shape(model.spectrogram_list))
    # print(model.spectrogram.T)
    # print(np.shape(model.spectrogram.T))
    # model.show_plot()