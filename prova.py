import matplotlib.pyplot as plt
import scipy.signal
import sklearn
from scipy import interpolate
import numpy as np

import numpy.fft as fft


import models

def first_test():
    original_list = models.utilities.get_questions_array()
    time_original = np.arange(0, len(original_list[0][0]))
    complete_x_list = models.utilities.get_questions_oversampled_array()
    time = np.arange(0, len(complete_x_list[0][0]))
    spectrum_orig = fft.fft(original_list)
    freq_orig = fft.fftfreq(len(spectrum_orig))
    spectrum = fft.fft(complete_x_list)
    freq = fft.fftfreq(len(spectrum))
    plt.plot(freq_orig, abs(spectrum_orig), '-b', freq, abs(spectrum), '-r')
    plt.show()
    print(np.mean(original_list[0][0]))
    print(np.mean(complete_x_list[0][0]))
    print(np.std(original_list[0][0]))
    print(np.std(complete_x_list[0][0]))
    plt.plot(time_original, original_list[0][0], '-b', time, complete_x_list[0][0], '-r')
    plt.savefig('overlapped.png')
    plt.plot(time_original, original_list[0][0], '-b')
    plt.savefig('original.png')
    plt.plot(time, complete_x_list[0][0], '-r')
    plt.savefig('new.png')


def second_test():

    time_original = np.arange(0, 53)
    original_list = np.sin(time_original)
    time = np.arange(0, 265)
    complete_x_list = sklearn.utils.resample(original_list, n_samples=265, stratify=original_list)
    # complete_x_list = scipy.signal.resample(original_list, 265)
    spectrum_orig = fft.fft(original_list)
    freq_orig = fft.fftfreq(len(spectrum_orig))
    spectrum = fft.fft(complete_x_list)
    freq = fft.fftfreq(len(spectrum))
    plt.plot(freq_orig, abs(spectrum_orig), '-b', freq, abs(spectrum), '-r')
    plt.show()
    print(complete_x_list.shape)
    print(np.mean(original_list))
    print(np.mean(complete_x_list))
    print(np.std(original_list))
    print(np.std(complete_x_list))
    plt.plot(time_original, original_list, '-b', time, complete_x_list, '-r')
    plt.savefig('overlapped.png')
    plt.plot(time_original, original_list, '-b')
    plt.savefig('original.png')
    plt.plot(time, complete_x_list, '-r')
    plt.savefig('new.png')


models.utilities.get_questions_interpolation_array()