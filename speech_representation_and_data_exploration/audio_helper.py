import pickle
import os
from os.path import join
import numpy as np
from scipy import signal
import librosa
import librosa.display


def load_samples_from_audios(path_to_dir='./data', file_extension='MP3'):
    sample_list = []
    filename_list = [f for f in os.listdir(join(path_to_dir, '')) if f.endswith(file_extension)]
    for filename in filename_list:
        samples, sample_rate = librosa.load(path_to_dir + filename)
        sample_data = (samples, sample_rate)
        sample_list.append(sample_data)
    return np.array(sample_list)


def trim_samples(sample_list, trim_step=1000, trim_ratio=0.1):
    trimmed_sample_list = []
    for samples, sample_rate in sample_list:
        threshold = np.std(samples) * trim_ratio
        start_idx = 0
        end_idx = len(samples)
        for i in range(0, len(samples), trim_step):
            if (np.std(samples[i:i+trim_step]) > threshold):
                start_idx = i
                break
        for i in range(len(samples), 0, -trim_step):
            if (np.std(samples[i-trim_step:i]) > threshold):
                end_idx = i
                break
        trimmed_sample_list.append((samples[start_idx:end_idx], sample_rate))
    return np.array(trimmed_sample_list)


def re_sampling(sample_list, duration=5):
    re_sample_list = []

    for samples, sample_rate in sample_list:
        sample_shape = samples.shape[0]
        target_shape = sample_rate * duration

        if target_shape < sample_shape:
            step_size = sample_shape // target_shape
            re_samples = [samples[i] for i in range(0, sample_shape, step_size)]
            re_samples = re_samples[:target_shape]
        elif target_shape > sample_shape:
            diff_shape = target_shape - sample_shape
            period = diff_shape // (sample_shape - 1)
            temp = []
            for i in range(len(samples) - 1):
                temp.extend(np.linspace(samples[i], samples[i + 1], period)[:-1])
            re_samples = temp.extend(samples[-1])
            re_samples = re_samples[:target_shape]
        else:
            re_samples = samples

        re_sample_list.append((np.array(re_samples), sample_rate))

    return np.array(re_sample_list)


def sample_to_spectrogram(sample_list, save_to_file=False, path_to_file=None):

    def log_specgram(audio, sample_rate, window_size=20,
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

    spectrogram_list = []
    for samples, sample_rate in sample_list:
        freqs, times, spectrogram = log_specgram(samples, sample_rate)
        # spectrogram = spectrogram.T
        spectrogram = np.expand_dims(spectrogram, axis=2)
        spectrogram_list.append(spectrogram)
    spectrogram_list = np.array(spectrogram_list)

    if save_to_file:
        if not path_to_file:
            print("no path")
            pass

        with open(path_to_file, 'wb') as _file:
            pickle.dump(spectrogram_list, _file)

    return spectrogram_list
