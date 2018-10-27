from playsound import playsound
import os
from os.path import isdir, join
from pathlib import Path
import pandas as pd

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
import librosa.display

import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
import pandas as pd



from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.callbacks import ModelCheckpoint, EarlyStopping


# mp3 to wav
# from pydub import AudioSegment  # ffmpeg 필요
# sound = AudioSegment.from_mp3(r'C:/Users/khudd/PycharmProjects/untitled/ml_study/data/001.MP3')
# sound.export("./data/001.wav", format="wav")

# playsound('./data/097.mp3')

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
        self.train_label = [[0,0,1],[0,1,0],[1,0,0]]

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
            self.samples, self.sample_rate = librosa.load(str(self.train_audio_path) + self.filename_list[i], duration=duration)
            self.freqs, self.times, self.spectrogram = self.log_specgram(self.samples, self.sample_rate)
            self.spectrogram_list.append(self.spectrogram.T)

        # self.samples, self.sample_rate = librosa.load(str(self.train_audio_path) + self.filename,
        #                                               duration=duration)
        # self.freqs, self.times, self.spectrogram = self.log_specgram(self.samples, self.sample_rate)
        # self.spectrogram_list.append(self.spectrogram)

    def re_sampling(self):
        div_num = 530496 // 22050
        re_samples = [self.samples[i] for i in range(530496) if i % div_num  == 0]
        re_samples = re_samples[:22050]
        return re_samples

    def show_plot(self):
        plt.figure()
        plt.imshow(self.spectrogram.T, aspect='auto', origin='lower',
                   extent=[self.times.min(), self.times.max(), self.freqs.min(), self.freqs.max()])
        plt.savefig('result.png')
        plt.show()

    def init_data(self):
        if os.path.isdir('./data'):
            waves = [f for f in os.listdir(join(self.train_audio_path, '')) if f.endswith('.wav')]
            self.filename_list = waves


    def trim_samples(self, samples=None, check_period=1000, ratio_threshold=0.1):
        if not samples:
            samples = self.samples

        threshold = np.std(samples) * ratio_threshold
        start_idx = 0
        end_idx = len(samples)

        for i in range(0, len(samples), check_period):
            if (np.std(samples[i:i + check_period]) > threshold):
                start_idx = i
                break

        for i in range(len(samples), 0, -check_period):
            if (np.std(samples[i - check_period:i]) > threshold):
                end_idx = i
                break

        return samples[start_idx:end_idx]

    def cnn(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(221, 497), activation='relu'))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=2))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(3, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])


if __name__=="__main__":
    model = SPEACH_PRO()
    model.init_data()
    model.set_sample(5)
    model.cnn()
    # model.trim_samples()
    print(np.shape(model.spectrogram_list[0]))
    model.model.fit(model.spectrogram_list[0],model.train_label[0])
    # print("np.shape(model.spectrogram_list)::",np.shape(model.spectrogram_list))
    # print(np.reshape(model.spectrogram_list,(221,497)))
    # print(np.shape(np.reshape(model.spectrogram_list,(221,497,1))))
    # print(model.spectrogram.T)
    # print(np.shape(model.spectrogram.T))
    # model.show_plot()


    # CNN
    # RNN





# fig = plt.figure(figsize=(14, 8))
# ax1 = fig.add_subplot(211)
# ax1.set_title('Raw wave of ' + filename)
# ax1.set_ylabel('Amplitude')
# ax1.plot(np.linspace(0, sample_rate/len(samples), sample_rate), samples)

# ax2 = fig.add_subplot(212)
# ax2.imshow(spectrogram.T, aspect='auto', origin='lower',
#            extent=[times.min(), times.max(), freqs.min(), freqs.max()])
# ax2.set_yticks(freqs[::16])
# ax2.set_xticks(times[::16])
# ax2.set_title('Spectrogram of ' + filename)
# ax2.set_ylabel('Freqs in Hz')
# ax2.set_xlabel('Seconds')
# plt.savefig()
# plt.show()


