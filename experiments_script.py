import numpy as np
import pandas as pd
import toml
from sklearn.model_selection import train_test_split

import models.deep_learning_models as dl_models

import models.deep_learning_models
from experiments.utilities import get_input_array_string, get_labels_array_string
import csv

config = toml.load('config.toml')

test_size_value = config['algorithm']['test_size']

f = open('results.csv', 'w')
f.write('index,model,label,input,loss_type,optimizer_type,dense_input,dense_input_dim,dense_input_activation,'
        'dense_output_activation,lstm_units,cnn_fiters,cnn_kernel,cnn_pool_size,dropout,dropout_value\n')
f.close()

f1 = open('performances.csv', 'w')
f1.write('index,acc,val_acc,loss,val_loss\n')
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
    f1 = open('performances.csv', 'a')
    line = str(c) + ',' + str(acc) + ',' + str(val_acc) + ',' + str(loss) + ',' + str(val_loss) + '\n'
    f1.write(line)
    f1.close()

def iterate_cnn1d(c, complete_x_list, complete_y_list):

    for dense_output_activation in config['algorithm']['activation_types']:
        for n_cnn_filters in config['algorithm']['n_cnn_filters']:
            for cnn_kernel_size in config['algorithm']['cnn_kernel_size']:
                for cnn_pool_size in config['algorithm']['cnn_pool_size']:
                    for dropout_value in config['algorithm']['dropout_value']:
                        for dense_input in config['general']['binary_value']:
                            if dense_input:
                                for dense_input_dim in config['algorithm']['dense_input_dim']:
                                    for dense_input_activation in config['algorithm']['activation_types']:

                                        X_train, X_test, y_train, y_test = train_test_split(complete_x_list,
                                                                                            complete_y_list,
                                                                                            test_size=test_size_value,
                                                                                            shuffle=True)

                                        X_train = np.array(X_train).astype(np.float32)
                                        X_test = np.asarray(X_test).astype(np.float32)

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

                                        history = model.fit(X_train, y_train, epochs=100,
                                                            validation_data=(X_test, y_test), shuffle=True)

                                        model.save('tf_models/test_model_' + str(c))

                                        history_dict = history.history

                                        name = 'CNN1D - ' + str(c)
                                        models.plots.plot_model_loss(history_dict, name)
                                        models.plots.plot_model_accuracy(history_dict, name)

                                        write_line_2(c, history_dict['accuracy'][-1],
                                                     history_dict['val_accuracy'][-1],
                                                     history_dict['loss'][-1],
                                                     history_dict['val_loss'][-1])

                                        c += 1
                            else:

                                X_train, X_test, y_train, y_test = train_test_split(complete_x_list, complete_y_list,
                                                                                    test_size=test_size_value,
                                                                                    shuffle=True)

                                X_train = np.array(X_train).astype(np.float32)
                                X_test = np.asarray(X_test).astype(np.float32)

                                model = dl_models.get_model_ccn1d(dense_input, 0, '',
                                                                  dense_output_activation, n_cnn_filters,
                                                                  cnn_kernel_size, cnn_pool_size, loss_type,
                                                                  optimizer_type, 1, dropout_value)

                                write_line(c, 'CNN1D', label_name, input_name,
                                           loss_type, optimizer_type, dense_input,
                                           '',
                                           '',
                                           dense_output_activation, '',
                                           n_cnn_filters,
                                           cnn_kernel_size, cnn_pool_size, 1,
                                           dropout_value)

                                history = model.fit(X_train, y_train, epochs=100,
                                                    validation_data=(X_test, y_test), shuffle=True)

                                model.save('tf_models/test_model_' + str(c))

                                history_dict = history.history

                                name = 'CNN1D - ' + str(c)
                                models.plots.plot_model_loss(history_dict, name)
                                models.plots.plot_model_accuracy(history_dict, name)

                                write_line_2(c, history_dict['accuracy'][-1],
                                             history_dict['val_accuracy'][-1],
                                             history_dict['loss'][-1],
                                             history_dict['val_loss'][-1])

                                c += 1

    return c

def iterate_lstm(c, complete_x_list, complete_y_list):

    for dense_output_activation in config['algorithm']['activation_types']:
        for n_lstm_units in config['algorithm']['n_lstm_units']:
            for dropout_value in config['algorithm']['dropout_value']:
                for dense_input in config['general']['binary_value']:

                    if dense_input:
                        for dense_input_dim in config['algorithm']['dense_input_dim']:
                            for dense_input_activation in config['algorithm']['activation_types']:
                                X_train, X_test, y_train, y_test = train_test_split(complete_x_list,
                                                                                    complete_y_list,
                                                                                    test_size=test_size_value,
                                                                                    shuffle=True)

                                X_train = np.array(X_train).astype(np.float32)
                                X_test = np.asarray(X_test).astype(np.float32)

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

                                history = model.fit(X_train, y_train, epochs=100,
                                                    validation_data=(X_test, y_test), shuffle=True)

                                model.save('tf_models/test_model_' + str(c))

                                history_dict = history.history

                                name = 'LSTM - ' + str(c)
                                models.plots.plot_model_loss(history_dict, name)
                                models.plots.plot_model_accuracy(history_dict, name)

                                write_line_2(c, history_dict['accuracy'][-1],
                                             history_dict['val_accuracy'][-1],
                                             history_dict['loss'][-1],
                                             history_dict['val_loss'][-1])

                                c += 1
                    else:

                        X_train, X_test, y_train, y_test = train_test_split(complete_x_list,
                                                                            complete_y_list,
                                                                            test_size=test_size_value,
                                                                            shuffle=True)

                        X_train = np.array(X_train).astype(np.float32)
                        X_test = np.asarray(X_test).astype(np.float32)

                        model = dl_models.get_model_lstm(dense_input, 0, '',
                                                         dense_output_activation,
                                                         n_lstm_units, loss_type, optimizer_type, 1,
                                                         dropout_value)

                        write_line(c, 'LSTM', label_name, input_name, loss_type,
                                   optimizer_type, dense_input, '', '',
                                   dense_output_activation, n_lstm_units, '', '', '',
                                   1, dropout_value)

                        history = model.fit(X_train, y_train, epochs=100,
                                            validation_data=(X_test, y_test), shuffle=True)

                        model.save('tf_models/test_model_' + str(c))

                        history_dict = history.history

                        name = 'LSTM - ' + str(c)
                        models.plots.plot_model_loss(history_dict, name)
                        models.plots.plot_model_accuracy(history_dict, name)

                        write_line_2(c, history_dict['accuracy'][-1],
                                     history_dict['val_accuracy'][-1],
                                     history_dict['loss'][-1],
                                     history_dict['val_loss'][-1])

                        c += 1

    return c


def iterate_cnn1d_lstm(c, complete_x_list, complete_y_list):

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

                                            X_train, X_test, y_train, y_test = train_test_split(complete_x_list,
                                                                                                complete_y_list,
                                                                                                test_size=test_size_value,
                                                                                                shuffle=True)

                                            X_train = np.array(X_train).astype(np.float32)
                                            X_test = np.asarray(X_test).astype(np.float32)

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
                                                       dense_output_activation, n_lstm_units,
                                                       n_cnn_filters,
                                                       cnn_kernel_size, cnn_pool_size,
                                                       1, dropout_value)

                                            history = model.fit(X_train, y_train, epochs=100,
                                                                validation_data=(X_test, y_test), shuffle=True)

                                            model.save('tf_models/test_model_' + str(c))

                                            history_dict = history.history

                                            name = 'CNN1D LSTM - ' + str(c)
                                            models.plots.plot_model_loss(history_dict, name)
                                            models.plots.plot_model_accuracy(history_dict, name)

                                            write_line_2(c, history_dict['accuracy'][-1],
                                                         history_dict['val_accuracy'][-1],
                                                         history_dict['loss'][-1],
                                                         history_dict['val_loss'][-1])

                                            c += 1
                                else:

                                    X_train, X_test, y_train, y_test = train_test_split(complete_x_list,
                                                                                        complete_y_list,
                                                                                        test_size=test_size_value,
                                                                                        shuffle=True)

                                    X_train = np.array(X_train).astype(np.float32)
                                    X_test = np.asarray(X_test).astype(np.float32)

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
                                               dense_output_activation, n_lstm_units,
                                               n_cnn_filters,
                                               cnn_kernel_size, cnn_pool_size,
                                               1,
                                               dropout_value)

                                    history = model.fit(X_train, y_train, epochs=100,
                                                        validation_data=(X_test, y_test), shuffle=True)

                                    model.save('tf_models/test_model_' + str(c))

                                    history_dict = history.history

                                    name = 'CNN1D LSTM - ' + str(c)
                                    models.plots.plot_model_loss(history_dict, name)
                                    models.plots.plot_model_accuracy(history_dict, name)

                                    write_line_2(c, history_dict['accuracy'][-1],
                                                 history_dict['val_accuracy'][-1],
                                                 history_dict['loss'][-1],
                                                 history_dict['val_loss'][-1])

                                    c += 1
    return c


def iterate_cnn1d_lstm_3dense(c, complete_x_list, complete_y_list):

    for dense_output_activation in config['algorithm']['activation_types']:
        for n_cnn_filters in config['algorithm']['n_cnn_filters']:
            for cnn_kernel_size in config['algorithm']['cnn_kernel_size']:
                for cnn_pool_size in config['algorithm']['cnn_pool_size']:
                    for n_lstm_units in config['algorithm']['n_lstm_units']:
                        for dropout_value in config['algorithm']['dropout_value']:
                            for dense_input in config['general']['binary_value']:

                                if dense_input:
                                    for dense_input_dim in config['algorithm']['dense_input_dim']:
                                        for dense_input_activation in config['algorithm']['activation_types']:

                                            X_train, X_test, y_train, y_test = train_test_split(complete_x_list,
                                                                                                complete_y_list,
                                                                                                test_size=test_size_value,
                                                                                                shuffle=True)

                                            X_train = np.array(X_train).astype(np.float32)
                                            X_test = np.asarray(X_test).astype(np.float32)

                                            model = dl_models.get_model_cnn1d_lstm_3x_dense(dense_input,
                                                                                   dense_input_dim,
                                                                                   dense_input_activation,
                                                                                   dense_output_activation,
                                                                                   n_cnn_filters,
                                                                                   cnn_kernel_size,
                                                                                   cnn_pool_size,
                                                                                   n_lstm_units, loss_type,
                                                                                   optimizer_type, 1,
                                                                                   dropout_value)

                                            write_line(c, 'CNN1D_LSTM_3DENSE', label_name,
                                                       input_name,
                                                       loss_type,
                                                       optimizer_type, dense_input,
                                                       dense_input_dim,
                                                       dense_input_activation,
                                                       dense_output_activation, n_lstm_units,
                                                       n_cnn_filters,
                                                       cnn_kernel_size, cnn_pool_size,
                                                       1, dropout_value)

                                            history = model.fit(X_train, y_train, epochs=100,
                                                                validation_data=(X_test, y_test), shuffle=True)

                                            model.save('tf_models/test_model_' + str(c))

                                            history_dict = history.history

                                            name = 'CNN1D LSTM 3DENSE - ' + str(c)
                                            models.plots.plot_model_loss(history_dict, name)
                                            models.plots.plot_model_accuracy(history_dict, name)

                                            write_line_2(c, history_dict['accuracy'][-1],
                                                         history_dict['val_accuracy'][-1],
                                                         history_dict['loss'][-1],
                                                         history_dict['val_loss'][-1])

                                            c += 1
                                else:

                                    X_train, X_test, y_train, y_test = train_test_split(complete_x_list,
                                                                                        complete_y_list,
                                                                                        test_size=test_size_value,
                                                                                        shuffle=True)

                                    X_train = np.array(X_train).astype(np.float32)
                                    X_test = np.asarray(X_test).astype(np.float32)

                                    model = dl_models.get_model_cnn1d_lstm_3x_dense(dense_input, 0, '',
                                                                           dense_output_activation,
                                                                           n_cnn_filters,
                                                                           cnn_kernel_size,
                                                                           cnn_pool_size,
                                                                           n_lstm_units,
                                                                           loss_type,
                                                                           optimizer_type, 1,
                                                                           dropout_value)

                                    write_line(c, 'CNN1D_LSTM_3DENSE', label_name,
                                               input_name,
                                               loss_type,
                                               optimizer_type, dense_input,
                                               '', '',
                                               dense_output_activation, n_lstm_units,
                                               n_cnn_filters,
                                               cnn_kernel_size, cnn_pool_size,
                                               1,
                                               dropout_value)

                                    history = model.fit(X_train, y_train, epochs=100,
                                                        validation_data=(X_test, y_test), shuffle=True)

                                    model.save('tf_models/test_model_' + str(c))

                                    history_dict = history.history

                                    name = 'CNN1D LSTM 3DENSE - ' + str(c)
                                    models.plots.plot_model_loss(history_dict, name)
                                    models.plots.plot_model_accuracy(history_dict, name)

                                    write_line_2(c, history_dict['accuracy'][-1],
                                                 history_dict['val_accuracy'][-1],
                                                 history_dict['loss'][-1],
                                                 history_dict['val_loss'][-1])

                                    c += 1
    return c


def iterate_2xcnn1d_lstm(c, complete_x_list, complete_y_list):

    for dense_output_activation in config['algorithm']['activation_types']:
        for n_cnn_filters in config['algorithm']['n_cnn_filters']:
            for cnn_kernel_size in config['algorithm']['cnn_kernel_size']:
                for cnn_pool_size in config['algorithm']['cnn_pool_size']:
                    for n_lstm_units in config['algorithm']['n_lstm_units']:
                        for dropout_value in config['algorithm']['dropout_value']:
                            for dense_input in config['general']['binary_value']:

                                if dense_input:
                                    for dense_input_dim in config['algorithm']['dense_input_dim']:
                                        for dense_input_activation in config['algorithm']['activation_types']:

                                            X_train, X_test, y_train, y_test = train_test_split(complete_x_list,
                                                                                                complete_y_list,
                                                                                                test_size=test_size_value,
                                                                                                shuffle=True)

                                            X_train = np.array(X_train).astype(np.float32)
                                            X_test = np.asarray(X_test).astype(np.float32)

                                            model = dl_models.get_model_2x_cnn1d_lstm(dense_input,
                                                                                   dense_input_dim,
                                                                                   dense_input_activation,
                                                                                   dense_output_activation,
                                                                                   n_cnn_filters,
                                                                                   cnn_kernel_size,
                                                                                   cnn_pool_size,
                                                                                   n_lstm_units, loss_type,
                                                                                   optimizer_type, 1,
                                                                                   dropout_value)

                                            write_line(c, '2xCNN1D_LSTM', label_name,
                                                       input_name,
                                                       loss_type,
                                                       optimizer_type, dense_input,
                                                       dense_input_dim,
                                                       dense_input_activation,
                                                       dense_output_activation, n_lstm_units,
                                                       n_cnn_filters,
                                                       cnn_kernel_size, cnn_pool_size,
                                                       1, dropout_value)

                                            history = model.fit(X_train, y_train, epochs=100,
                                                                validation_data=(X_test, y_test), shuffle=True)

                                            model.save('tf_models/test_model_' + str(c))

                                            history_dict = history.history

                                            name = '2CNN1D LSTM - ' + str(c)
                                            models.plots.plot_model_loss(history_dict, name)
                                            models.plots.plot_model_accuracy(history_dict, name)

                                            write_line_2(c, history_dict['accuracy'][-1],
                                                         history_dict['val_accuracy'][-1],
                                                         history_dict['loss'][-1],
                                                         history_dict['val_loss'][-1])

                                            c += 1
                                else:

                                    X_train, X_test, y_train, y_test = train_test_split(complete_x_list,
                                                                                        complete_y_list,
                                                                                        test_size=test_size_value,
                                                                                        shuffle=True)

                                    X_train = np.array(X_train).astype(np.float32)
                                    X_test = np.asarray(X_test).astype(np.float32)

                                    model = dl_models.get_model_2x_cnn1d_lstm(dense_input, 0, '',
                                                                           dense_output_activation,
                                                                           n_cnn_filters,
                                                                           cnn_kernel_size,
                                                                           cnn_pool_size,
                                                                           n_lstm_units,
                                                                           loss_type,
                                                                           optimizer_type, 1,
                                                                           dropout_value)

                                    write_line(c, '2xCNN1D_LSTM_3DENSE', label_name,
                                               input_name,
                                               loss_type,
                                               optimizer_type, dense_input,
                                               '', '',
                                               dense_output_activation, n_lstm_units,
                                               n_cnn_filters,
                                               cnn_kernel_size, cnn_pool_size,
                                               1,
                                               dropout_value)

                                    history = model.fit(X_train, y_train, epochs=100,
                                                        validation_data=(X_test, y_test), shuffle=True)

                                    model.save('tf_models/test_model_' + str(c))

                                    history_dict = history.history

                                    name = '2CNN1D LSTM - ' + str(c)
                                    models.plots.plot_model_loss(history_dict, name)
                                    models.plots.plot_model_accuracy(history_dict, name)

                                    write_line_2(c, history_dict['accuracy'][-1],
                                                 history_dict['val_accuracy'][-1],
                                                 history_dict['loss'][-1],
                                                 history_dict['val_loss'][-1])

                                    c += 1
    return c


def iterate_cnn2d(c, complete_x_list, complete_y_list):
    complete_x_list = np.expand_dims(complete_x_list, 2)
    complete_y_list = np.expand_dims(complete_y_list, 2)

    for dense_output_activation in config['algorithm']['activation_types']:
        for n_cnn_filters in config['algorithm']['n_cnn_filters']:
            for cnn_kernel_size in config['algorithm']['cnn_kernel_size']:
                for cnn_pool_size in config['algorithm']['cnn_pool_size']:
                    for dropout_value in config['algorithm']['dropout_value']:
                        for dense_input in config['general']['binary_value']:

                            if dense_input:
                                for dense_input_dim in config['algorithm']['dense_input_dim']:
                                    for dense_input_activation in config['algorithm']['activation_types']:

                                        X_train, X_test, y_train, y_test = train_test_split(complete_x_list,
                                                                                            complete_y_list,
                                                                                            test_size=test_size_value,
                                                                                            shuffle=True)

                                        X_train = np.array(X_train).astype(np.float32)
                                        X_test = np.asarray(X_test).astype(np.float32)

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

                                        history = model.fit(X_train, y_train, epochs=100,
                                                            validation_data=(X_test, y_test), shuffle=True)

                                        model.save('tf_models/test_model_' + str(c))

                                        history_dict = history.history

                                        name = 'CNN2D - ' + str(c)
                                        models.plots.plot_model_loss(history_dict, name)
                                        models.plots.plot_model_accuracy(history_dict, name)

                                        write_line_2(c, history_dict['accuracy'][-1],
                                                     history_dict['val_accuracy'][-1],
                                                     history_dict['loss'][-1],
                                                     history_dict['val_loss'][-1])

                                        c += 1
                            else:

                                X_train, X_test, y_train, y_test = train_test_split(complete_x_list, complete_y_list,
                                                                                    test_size=test_size_value, shuffle=True)

                                X_train = np.array(X_train).astype(np.float32)
                                X_test = np.asarray(X_test).astype(np.float32)

                                model = dl_models.get_model_cnn2d(dense_input, 0, '', dense_output_activation,
                                                                  n_cnn_filters, cnn_kernel_size, cnn_pool_size,
                                                                  loss_type, optimizer_type, 1, dropout_value)

                                write_line(c, 'CNN2D', label_name, input_name, loss_type, optimizer_type, dense_input,
                                           '', '', dense_output_activation, '', n_cnn_filters, cnn_kernel_size,
                                           cnn_pool_size, 1, dropout_value)

                                history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test),
                                                    shuffle=True)

                                model.save('tf_models/test_model_' + str(c))

                                history_dict = history.history

                                name = 'CNN2D - ' + str(c)
                                models.plots.plot_model_loss(history_dict, name)
                                models.plots.plot_model_accuracy(history_dict, name)

                                write_line_2(c, history_dict['accuracy'][-1],
                                             history_dict['val_accuracy'][-1],
                                             history_dict['loss'][-1],
                                             history_dict['val_loss'][-1])

    return c


def iterate_cnn2d_lstm(c, complete_x_list, complete_y_list):

    complete_x_list = np.expand_dims(complete_x_list, 2)
    complete_y_list = np.expand_dims(complete_y_list, 2)

    for dense_output_activation in config['algorithm']['activation_types']:
        for n_cnn_filters in config['algorithm']['n_cnn_filters']:
            for cnn_kernel_size in config['algorithm']['cnn_kernel_size']:
                for cnn_pool_size in config['algorithm']['cnn_pool_size']:
                    for n_lstm_units in config['algorithm']['n_lstm_units']:
                        for dropout_value in config['algorithm']['dropout_value']:
                            for dense_input in config['general']['binary_value']:

                                if dense_input:
                                    for dense_input_dim in config['algorithm']['dense_input_dim']:
                                        for dense_input_activation in config['algorithm']['activation_types']:

                                            X_train, X_test, y_train, y_test = train_test_split(complete_x_list,
                                                                                                complete_y_list,
                                                                                                test_size=test_size_value,
                                                                                                shuffle=True)

                                            X_train = np.array(X_train).astype(np.float32)
                                            X_test = np.asarray(X_test).astype(np.float32)

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
                                                       dense_output_activation, n_lstm_units,
                                                       n_cnn_filters,
                                                       cnn_kernel_size, cnn_pool_size, 1,
                                                       dropout_value)

                                            history = model.fit(X_train, y_train, epochs=100,
                                                                validation_data=(X_test, y_test), shuffle=True)

                                            model.save('tf_models/test_model_' + str(c))

                                            history_dict = history.history

                                            name = 'CNN2D LSTM - ' + str(c)
                                            models.plots.plot_model_loss(history_dict, name)
                                            models.plots.plot_model_accuracy(history_dict, name)

                                            write_line_2(c, history_dict['accuracy'][-1],
                                                         history_dict['val_accuracy'][-1],
                                                         history_dict['loss'][-1],
                                                         history_dict['val_loss'][-1])

                                            c += 1
                                else:

                                    X_train, X_test, y_train, y_test = train_test_split(complete_x_list,
                                                                                        complete_y_list,
                                                                                        test_size=test_size_value,
                                                                                        shuffle=True)

                                    X_train = np.array(X_train).astype(np.float32)
                                    X_test = np.asarray(X_test).astype(np.float32)

                                    model = dl_models.get_model_cnn2d_lstm(dense_input, 0, n_lstm_units,
                                                                           dense_output_activation,
                                                                           n_cnn_filters, cnn_kernel_size,
                                                                           cnn_pool_size,
                                                                           n_lstm_units, loss_type,
                                                                           optimizer_type, dropout_value)

                                    write_line(c, 'CNN2D_LSTM', label_name, input_name,
                                               loss_type,
                                               optimizer_type, dense_input,
                                               '', '',
                                               dense_output_activation, n_lstm_units, n_cnn_filters,
                                               cnn_kernel_size, cnn_pool_size, 1,
                                               dropout_value)

                                    history = model.fit(X_train, y_train, epochs=100,
                                                        validation_data=(X_test, y_test), shuffle=True)

                                    model.save('tf_models/test_model_' + str(c))

                                    history_dict = history.history

                                    name = 'CNN2D LSTM - ' + str(c)
                                    models.plots.plot_model_loss(history_dict, name)
                                    models.plots.plot_model_accuracy(history_dict, name)

                                    write_line_2(c, history_dict['accuracy'][-1],
                                                 history_dict['val_accuracy'][-1],
                                                 history_dict['loss'][-1],
                                                 history_dict['val_loss'][-1])

                                    c += 1

    return c

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

                c = iterate_cnn1d_lstm(c, complete_x_list, complete_y_list)
                c = iterate_cnn1d_lstm_3dense(c, complete_x_list, complete_y_list)
                c = iterate_2xcnn1d_lstm(c, complete_x_list, complete_y_list)
                c = iterate_lstm(c, complete_x_list, complete_y_list)
                c = iterate_cnn1d(c, complete_x_list, complete_y_list)
                c = iterate_cnn2d_lstm(c, complete_x_list, complete_y_list)
                c = iterate_cnn2d(c, complete_x_list, complete_y_list)