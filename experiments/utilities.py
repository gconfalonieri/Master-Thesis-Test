import os
import random
import tensorflow as tf
import numpy as np
import toml

config = toml.load('config.toml')

filename_results = config['path']['filename_results']
filename_performances_cross_corr = config['path']['filename_performances_cross_corr']
filename_performances_aggregate = config['path']['filename_performances_aggregate']

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


def init_files():

    f = open(filename_results, 'w')
    f.write('index,model,label,input,loss_type,optimizer_type,dense_input,dense_input_dim,dense_input_activation,'
            'dense_output_activation,lstm_units,cnn_fiters,cnn_kernel,cnn_pool_size,dropout,dropout_value\n')
    f.close()

    f = open(filename_performances_cross_corr, 'w')
    f.write('index,cross_step,acc,val_acc,loss,val_loss\n')
    f.close()

    f = open(filename_performances_aggregate, 'w')
    f.write('index,acc,val_acc,loss,val_loss\n')
    f.close()


def write_line_results(filename, c, model, label_name, input_name, loss_type, optimizer_type, dense_input, dense_input_dim,
               dense_input_activation, dense_output_activation, lstm_cells, n_cnn_filters, cnn_kernel_size,
               cnn_pool_size, dropout, dropout_value):
    f = open(filename, 'a')
    line = str(c) + ',' + model + ',' + label_name + ',' + input_name \
           + ',' + loss_type + ',' + optimizer_type + ',' + str(dense_input) \
           + ',' + str(dense_input_dim) + ',' + dense_input_activation + ',' \
           + dense_output_activation + ',' + str(lstm_cells) + ',' + str(n_cnn_filters) \
           + ',' + str(cnn_kernel_size) + ',' + str(cnn_pool_size) + ',' + str(dropout) \
           + ',' + str(dropout_value) + '\n'
    f.write(line)
    f.close()


def write_line_performances(filename, c, acc, val_acc, loss, val_loss):
    f1 = open(filename, 'a')
    line = str(c) + ',' + str(acc) + ',' + str(val_acc) + ',' + str(loss) + ',' + str(val_loss) + '\n'
    f1.write(line)
    f1.close()


def write_line_performances_cross_correlation(filename, c, cross_counter, acc, val_acc, loss, val_loss):
    f1 = open(filename, 'a')
    line = str(c) + ',' + str(cross_counter) + ',' + str(acc) + ',' + str(val_acc) + ',' + str(loss) + ',' + str(val_loss) + '\n'
    f1.write(line)
    f1.close()