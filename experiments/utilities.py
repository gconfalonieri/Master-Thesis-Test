import os
import random
import tensorflow as tf
import numpy as np
import toml

config = toml.load('config.toml')

def get_input_array_string(input_array):
    name = ''
    if input_array == 'datasets/arrays/undersampled/input_1_1_oversampled.npy' or input_array == 'datasets/arrays/undersampled_shifted/input_1_1_oversampled.npy':
        name = '-1_1_OVERSAMPLED'
    elif input_array == 'datasets/arrays/undersampled/input_1_1_padded_begin.npy':
        name = '-1_1_PADDED_BEGIN'
    elif input_array == 'datasets/arrays/undersampled/input_1_1_padded_end.npy':
        name = '-1_1_PADDED_END'
    elif input_array == 'datasets/arrays/undersampled/input_0_1_oversampled.npy':
        name = '0_1_OVERSAMPLED'
    elif input_array ==  'datasets/arrays/undersampled/input_0_1_padded_begin.npy':
        name = '0_1_PADDED_BEGIN'
    elif input_array == 'datasets/arrays/undersampled/input_0_1_padded_end.npy':
        name = '0_1_PADDED_END'

    return name


def get_labels_array_string(labels_array):
    name = ''
    if labels_array == 'datasets/arrays/labels/labels_v2.npy':
        name = 'TIMES_ONLY_V2'
    elif labels_array == 'datasets/arrays/labels/labels_v2_2F.npy':
        name = 'TIMES_ONLY_V2_2F'

    return name


def fix_seeds():
    os.environ['PYTHONHASHSEED'] = config['random_seed']['pythonhashseed']
    random.seed(config['random_seed']['python_seed'])
    np.random.seed(config['random_seed']['numpy_seed'])
    tf.random.set_seed(config['random_seed']['tf_seed'])