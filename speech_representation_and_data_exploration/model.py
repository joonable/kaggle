import numpy as np
import pickle
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from os.path import isdir, join
from sklearn.model_selection import train_test_split
import sys

import audio_helper as ah

class CNN(object):

    def __init__(self, config):
        self.train_audio_path = './data/'
        self.config = {}    #TODO config lib
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.cnn = None
        self.num_class = None

    def load_dataset(self, split_data=False):
        with open(self.config['data_path'], 'rb') as _file:
            x = pickle.loads(_file)
            y = np.diag(np.eye(len(x)))     #TODO categorise y_values

        self.input_shape = x[0].shape
        self.num_class = y.shape[1]

        if split_data:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2, random_state=12)
        else:
            self.x_train, self.y_train = x, y

    def model(self):
        self.cnn = Sequential()
        self.cnn.add(Conv2D(32, kernel_size=(3, 3), input_shape=self.input_shape, activation='relu'))
        self.cnn.add(Conv2D(64, (3, 3), activation='relu'))
        self.cnn.add(MaxPooling2D(pool_size=2))
        self.cnn.add(Dropout(0.25))
        self.cnn.add(Flatten())
        self.cnn.add(Dense(128, activation='relu'))
        self.cnn.add(Dropout(0.5))
        self.cnn.add(Dense(2, activation='softmax'))
        self.cnn.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])


    def feed_test(self):
        pass

    def train(self):
        #TODO train_code
        pass

    def predict(self):
        # TODO test_code
        pass


if __name__=="__main__":
    sample_list = ah.load_samples_from_audios(path='./data/', file_extension='MP3')
    sample_list = ah.trim_samples(sample_list, trim_step=1000, trim_ratio=0.1)
    sample_list = ah.re_sampling(sample_list, duration=5)
    spectrogram_list = ah.sample_to_spectrogram(sample_list, save_to_file=True, path_to_file='./data/data.pkl')
    print(spectrogram_list.shape)
    # print(len(spectrogram_list))
    config = {}
    cnn = CNN(config=config)
# model = SPEACH_PRO()
#     model.set_sample(5)
#     model.cnn()
#     model.trim_samples()
#     print(np.shape(model.spectrogram_list[0]))
    # print(np.shape(model.spectrogram_list))
    # model.model.fit(np.array(model.spectrogram_list), np.array(model.train_label))

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