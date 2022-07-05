import numpy as np
import pandas as pd
import toml
import models.deep_learning_models as dl_models

import models.deep_learning_models
from experiments.utilities import get_input_array_string, get_labels_array_string
import csv

f = open('results.csv', 'w')
f.write('index,model,label,input,loss_type,optimizer_type,dense_input,dense_input_dim,dense_input_activation,'
        'dense_output_activation,lstm_units,cnn_fiters,cnn_kernel,cnn_pool_size,dropout,dropout_value')
f.close()

f1 = open('performaces.csv', 'w')
f1.write('index,acc,val_acc,loss,val_loss')
f1.close()

def write_line(c, model, label_name, input_name, loss_type, optimizer_type, dense_input, dense_input_dim,
               dense_input_activation, dense_output_activation, lstm_cells, n_cnn_filters, cnn_kernel_size,
               cnn_pool_size, dropout, dropout_value):
    f = open('results.csv', 'a')
    line = str(c) + ',' + model + ',' + label_name + ',' + input_name \
           + ',' + loss_type + ',' + optimizer_type + ',' + str(dense_input) \
           + ',' + str(dense_input_dim) + ',' + dense_input_activation + ',' \
           + dense_output_activation + ',' + str(lstm_cells) + ',' + str(n_cnn_filters) \
           + ',' + str(cnn_kernel_size) + ',' + str(cnn_pool_size) + ',' + str(dropout) \
           + ',' + str(dropout_value) + '\n'
    f.write(line)
    f.close()


def write_line_2(c, acc, val_acc, loss, val_loss):
    f1 = open('performaces.csv', 'a')
    line = str(c) + ',' + str(acc) + ',' + str(val_acc) + ',' + str(loss) + ',' + str(val_loss) + '\n'
    f1.write(line)
    f1.close()

config = toml.load('config.toml')

performances_df = pd.DataFrame(columns=['index', 'acc', 'val_acc', 'loss', 'val_loss'])

c = 0

for label_array in config['path']['labels_arrays']:

    complete_y_list = np.load(label_array, allow_pickle=True)
    label_name = get_labels_array_string(label_array)

    for input_array in config['path']['input_arrays']:

        complete_x_list = np.load(input_array, allow_pickle=True)
        input_name = get_input_array_string(input_array)

        for loss_type in config['algorithm']['loss_types']:
            for optimizer_type in config['algorithm']['optimizer_types']:

                # CNN 1D
                for dense_output_activation in config['algorithm']['activation_types']:
                    for n_cnn_filters in config['algorithm']['n_cnn_filters']:
                        for cnn_kernel_size in config['algorithm']['cnn_kernel_size']:
                            for cnn_pool_size in config['algorithm']['cnn_pool_size']:
                                for dropout_value in config['algorithm']['dropout_value']:
                                    for dense_input in config['general']['binary_value']:
                                        if dense_input:
                                            for dense_input_dim in config['algorithm']['dense_input_dim']:
                                                for dense_input_activation in config['algorithm']['activation_types']:
                                                    model = dl_models.get_model_ccn1d(dense_input, dense_input_dim,
                                                                                      dense_input_activation,
                                                                                      dense_output_activation,
                                                                                      n_cnn_filters,
                                                                                      cnn_kernel_size, cnn_pool_size,
                                                                                      loss_type,
                                                                                      optimizer_type, 1, dropout_value)

                                                    write_line(c, 'CNN1D', label_name, input_name,
                                                               loss_type, optimizer_type, dense_input,
                                                               dense_input_dim,
                                                               dense_input_activation,
                                                               dense_output_activation, '',
                                                               n_cnn_filters,
                                                               cnn_kernel_size, cnn_pool_size, 1,
                                                               dropout_value)

                                                    c += 1
                                        else:
                                            model = dl_models.get_model_ccn1d(dense_input, 0, '',
                                                                              dense_output_activation, n_cnn_filters,
                                                                              cnn_kernel_size, cnn_pool_size, loss_type,
                                                                              optimizer_type, 1, dropout_value)

                                            write_line(c, 'CNN1D', label_name, input_name,
                                                       loss_type, optimizer_type, dense_input,
                                                       '',
                                                       '',
                                                       dense_output_activation, 'NULL',
                                                       n_cnn_filters,
                                                       cnn_kernel_size, cnn_pool_size, 1,
                                                       dropout_value)

                                            c += 1

                # LSTM
                for dense_output_activation in config['algorithm']['activation_types']:
                    for n_lstm_units in config['algorithm']['n_lstm_units']:
                        for dropout_value in config['algorithm']['dropout_value']:
                            for dense_input in config['general']['binary_value']:

                                if dense_input:
                                    for dense_input_dim in config['algorithm']['dense_input_dim']:
                                        for dense_input_activation in config['algorithm']['activation_types']:
                                            model = dl_models.get_model_lstm(dense_input, dense_input_dim,
                                                                             dense_input_activation,
                                                                             dense_output_activation,
                                                                             n_lstm_units, loss_type,
                                                                             optimizer_type, 1, dropout_value)

                                            write_line(c, 'LSTM', label_name, input_name,
                                                       loss_type,
                                                       optimizer_type, dense_input, dense_input_dim,
                                                       dense_input_activation,
                                                       dense_output_activation, n_lstm_units, '',
                                                       '', '', 1,
                                                       dropout_value)

                                            c += 1
                                else:
                                    model = dl_models.get_model_lstm(dense_input, 0, '',
                                                                     dense_output_activation,
                                                                     n_lstm_units, loss_type, optimizer_type, 1,
                                                                     dropout_value)

                                    write_line(c, 'LSTM', label_name, input_name, loss_type,
                                               optimizer_type, dense_input, '', '',
                                               dense_output_activation, n_lstm_units, '', '', '',
                                               1, dropout_value)

                                    c += 1

                # CNN1D + LSTM
                for dense_output_activation in config['algorithm']['activation_types']:
                    for n_cnn_filters in config['algorithm']['n_cnn_filters']:
                        for cnn_kernel_size in config['algorithm']['cnn_kernel_size']:
                            for cnn_pool_size in config['algorithm']['cnn_pool_size']:
                                for n_lstm_units in config['algorithm']['n_lstm_units']:
                                    for dropout_value in config['algorithm']['dropout_value']:
                                        for dense_input in config['general']['binary_value']:

                                            if dense_input:
                                                for dense_input_dim in config['algorithm']['dense_input_dim']:
                                                    for dense_input_activation in config['algorithm'][
                                                        'activation_types']:
                                                        model = dl_models.get_model_cnn1d_lstm(dense_input,
                                                                                               dense_input_dim,
                                                                                               dense_input_activation,
                                                                                               dense_output_activation,
                                                                                               n_cnn_filters,
                                                                                               cnn_kernel_size,
                                                                                               cnn_pool_size,
                                                                                               n_lstm_units, loss_type,
                                                                                               optimizer_type, 1,
                                                                                               dropout_value)

                                                        write_line(c, 'CNN1D_LSTM', label_name,
                                                                   input_name,
                                                                   loss_type,
                                                                   optimizer_type, dense_input,
                                                                   dense_input_dim,
                                                                   dense_input_activation,
                                                                   dense_output_activation, '',
                                                                   n_cnn_filters,
                                                                   cnn_kernel_size, cnn_pool_size,
                                                                   1, dropout_value)

                                                        c += 1
                                            else:
                                                model = dl_models.get_model_cnn1d_lstm(dense_input, 0, '',
                                                                                       dense_output_activation,
                                                                                       n_cnn_filters,
                                                                                       cnn_kernel_size,
                                                                                       cnn_pool_size,
                                                                                       n_lstm_units,
                                                                                       loss_type,
                                                                                       optimizer_type, 1,
                                                                                       dropout_value)

                                                write_line(c, 'CNN1D_LSTM', label_name,
                                                           input_name,
                                                           loss_type,
                                                           optimizer_type, dense_input,
                                                           '', '',
                                                           dense_output_activation, '',
                                                           n_cnn_filters,
                                                           cnn_kernel_size, cnn_pool_size,
                                                           1,
                                                           dropout_value)

                                                c += 1

                # CNN2D
                for dense_output_activation in config['algorithm']['activation_types']:
                    for n_cnn_filters in config['algorithm']['n_cnn_filters']:
                        for cnn_kernel_size in config['algorithm']['cnn_kernel_size']:
                            for cnn_pool_size in config['algorithm']['cnn_pool_size']:
                                for dropout_value in config['algorithm']['dropout_value']:
                                    for dense_input in config['general']['binary_value']:

                                        if dense_input:
                                            for dense_input_dim in config['algorithm']['dense_input_dim']:
                                                for dense_input_activation in config['algorithm']['activation_types']:
                                                    model = dl_models.get_model_cnn2d(dense_input,
                                                                                      dense_input_dim,
                                                                                      dense_input_activation,
                                                                                      dense_output_activation,
                                                                                      n_cnn_filters,
                                                                                      cnn_kernel_size, cnn_pool_size,
                                                                                      loss_type,
                                                                                      optimizer_type, 1, dropout_value)

                                                    write_line(c, 'CNN2D', label_name, input_name,
                                                               loss_type,
                                                               optimizer_type, dense_input,
                                                               dense_input_dim,
                                                               dense_input_activation,
                                                               dense_output_activation, '',
                                                               n_cnn_filters,
                                                               cnn_kernel_size, cnn_pool_size, 1,
                                                               dropout_value)

                                                    c += 1
                                        else:
                                            model = dl_models.get_model_cnn2d(dense_input,
                                                                              0, '',
                                                                              dense_output_activation, n_cnn_filters,
                                                                              cnn_kernel_size, cnn_pool_size, loss_type,
                                                                              optimizer_type, 1, dropout_value)

                                            write_line(c, 'CNN2D', label_name, input_name,
                                                       loss_type,
                                                       optimizer_type, dense_input, '', '',
                                                       dense_output_activation, '', n_cnn_filters,
                                                       cnn_kernel_size, cnn_pool_size, 1,
                                                       dropout_value)

                # CNN2D + LSTM
                for dense_output_activation in config['algorithm']['activation_types']:
                    for n_cnn_filters in config['algorithm']['n_cnn_filters']:
                        for cnn_kernel_size in config['algorithm']['cnn_kernel_size']:
                            for cnn_pool_size in config['algorithm']['cnn_pool_size']:
                                for n_lstm_units in config['algorithm']['n_lstm_units']:
                                    for dropout_value in config['algorithm']['dropout_value']:
                                        for dense_input in config['general']['binary_value']:

                                            if dense_input:
                                                for dense_input_dim in config['algorithm']['dense_input_dim']:
                                                    for dense_input_activation in config['algorithm'][
                                                        'activation_types']:
                                                        model = dl_models.get_model_cnn2d_lstm(dense_input,
                                                                                               dense_input_dim,
                                                                                               dense_input_activation,
                                                                                               dense_output_activation,
                                                                                               n_cnn_filters,
                                                                                               cnn_kernel_size,
                                                                                               cnn_pool_size,
                                                                                               n_lstm_units, loss_type,
                                                                                               optimizer_type,
                                                                                               dropout_value)

                                                        write_line(c, 'CNN2D_LSTM', label_name,
                                                                   input_name,
                                                                   loss_type,
                                                                   optimizer_type, dense_input,
                                                                   dense_input_dim,
                                                                   dense_input_activation,
                                                                   dense_output_activation, '',
                                                                   n_cnn_filters,
                                                                   cnn_kernel_size, cnn_pool_size, 1,
                                                                   dropout_value)

                                                        c += 1
                                            else:
                                                model = dl_models.get_model_cnn2d_lstm(dense_input, 0, '',
                                                                                       dense_output_activation,
                                                                                       n_cnn_filters, cnn_kernel_size,
                                                                                       cnn_pool_size,
                                                                                       n_lstm_units, loss_type,
                                                                                       optimizer_type, dropout_value)

                                                write_line(c, 'CNN2D_LSTM', label_name, input_name,
                                                            loss_type,
                                                            optimizer_type, dense_input,
                                                            '', '',
                                                            dense_output_activation, '', n_cnn_filters,
                                                            cnn_kernel_size, cnn_pool_size, 1,
                                                            dropout_value)

                                                c += 1
