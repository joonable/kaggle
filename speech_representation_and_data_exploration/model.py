import numpy as np
import pickle
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from sklearn.model_selection import train_test_split
import audio_helper as ah


class CNN(object):

    def __init__(self, config):
        # self.train_audio_path = './data/'
        self.config = config    #TODO config lib

        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

        self.num_class = None
        self.input_shape = None
        self._cnn = None

    def load_dataset(self, split_data=False):
        with open(self.config['data_path'], 'rb') as _file:
            # x = pickle.loads(_file)
            x = pickle.load(_file)
            y = np.eye(x.shape[0])     #TODO categorise y_values

        self.input_shape = x[0].shape
        self.num_class = y.shape[1]

        if split_data:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2, random_state=12)
        else:
            self.x_train, self.y_train = x, y

    def model(self):
        self._cnn = Sequential()
        self._cnn.add(Conv2D(32, kernel_size=(3, 3), input_shape=self.input_shape, activation='relu'))
        self._cnn.add(Conv2D(64, (3, 3), activation='relu'))
        self._cnn.add(MaxPooling2D(pool_size=2))
        self._cnn.add(Dropout(0.25))
        self._cnn.add(Flatten())
        self._cnn.add(Dense(128, activation='relu'))
        self._cnn.add(Dropout(0.5))
        self._cnn.add(Dense(2, activation='softmax'))
        self._cnn.compile(loss='categorical_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])


    def feed_test(self):
        self._cnn.fit(self.x_train, self.y_train)

    def train(self):
        #TODO train_code
        pass

    def predict(self):
        # TODO predict_code
        pass


if __name__=="__main__":
    sample_list = ah.load_samples_from_audios(path_to_dir='./data/', file_extension='MP3')
    sample_list = ah.trim_samples(sample_list, trim_step=1000, trim_ratio=0.1)
    sample_list = ah.re_sampling(sample_list, duration=5)
    spectrogram_list = ah.sample_to_spectrogram(sample_list, save_to_file=True, path_to_file='./data/data.pkl')

    config = {"data_path": './data/data.pkl'}       #TODO read_from_file
    cnn = CNN(config=config)
    cnn.load_dataset(split_data=False)
    cnn.model()
    cnn.feed_test()
