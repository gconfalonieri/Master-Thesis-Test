import numpy as np
import toml
from sklearn.model_selection import KFold
import experiments.utilities
import models
import models.deep_learning_models as dl_models

config = toml.load('config.toml')

experiments.utilities.fix_seeds()

filename_results = config['path']['filename_results']
filename_performances_cross_corr = config['path']['filename_performances_cross_corr']
filename_performances_aggregate = config['path']['filename_performances_aggregate']

n_split = config['algorithm']['n_kfold_splits']


def cross_validation_users(c, model, array_x, array_y, model_name):
    cross_counter = 0
    acc_list = []
    val_acc_list = []
    loss_list = []
    val_loss_list = []
    for train_index, test_index in KFold(n_split).split(array_x):
        x_train, x_test = models.utilities.aggregate_users(array_x[train_index]), models.utilities.aggregate_users(
            array_x[test_index])
        y_train, y_test = models.utilities.aggregate_users_labels(
            array_y[train_index]), models.utilities.aggregate_users_labels(array_y[test_index])
        history = model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test), shuffle=True)
        history_dict = history.history
        experiments.utilities.write_line_performances_cross_correlation(filename_performances_cross_corr, c,
                                                                        cross_counter,
                                                                        history_dict['accuracy'][-1],
                                                                        history_dict['val_accuracy'][-1],
                                                                        history_dict['loss'][-1],
                                                                        history_dict['val_loss'][-1])
        acc_list.append(history_dict['accuracy'][-1])
        val_acc_list.append(history_dict['val_accuracy'][-1])
        loss_list.append(history_dict['loss'][-1])
        val_loss_list.append(history_dict['val_loss'][-1])
        name = model_name + ' - ' + str(c) + ' - ' + str(cross_counter)
        models.plots.plot_model_loss(history_dict, name)
        models.plots.plot_model_accuracy(history_dict, name)
        cross_counter += 1
    experiments.utilities.write_line_performances(filename_performances_aggregate, c, np.mean(acc_list),
                                                  np.mean(val_acc_list), np.mean(loss_list), np.mean(val_loss_list))


def iterate_cnn1d(c, x_array, y_array, loss_type, optimizer_type, label_name, input_name):
    model_name = 'CNN1D'

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

                                        experiments.utilities.write_line_results(filename_results, c, model_name,
                                                                                 label_name,
                                                                                 input_name,
                                                                                 loss_type, optimizer_type,
                                                                                 dense_input,
                                                                                 dense_input_dim,
                                                                                 dense_input_activation,
                                                                                 dense_output_activation, '',
                                                                                 n_cnn_filters,
                                                                                 cnn_kernel_size, cnn_pool_size, 1,
                                                                                 dropout_value)

                                        cross_validation_users(c, model, x_array, y_array, model_name)

                                        # model.save('tf_models/test_model_' + str(c))

                                        c += 1
                            else:

                                model = dl_models.get_model_ccn1d(dense_input, 0, '',
                                                                  dense_output_activation, n_cnn_filters,
                                                                  cnn_kernel_size, cnn_pool_size, loss_type,
                                                                  optimizer_type, 1, dropout_value)

                                experiments.utilities.write_line_results(filename_results, c, model_name, label_name,
                                                                         input_name,
                                                                         loss_type, optimizer_type, dense_input,
                                                                         '',
                                                                         '',
                                                                         dense_output_activation, '',
                                                                         n_cnn_filters,
                                                                         cnn_kernel_size, cnn_pool_size, 1,
                                                                         dropout_value)

                                cross_validation_users(c, model, x_array, y_array, model_name)

                                c += 1

    return c


def iterate_lstm(c, x_array, y_array, loss_type, optimizer_type, label_name, input_name):
    model_name = 'LSTM'

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

                                experiments.utilities.write_line_results(filename_results, c, model_name, label_name,
                                                                         input_name, loss_type,
                                                                         optimizer_type, dense_input, dense_input_dim,
                                                                         dense_input_activation,
                                                                         dense_output_activation, n_lstm_units, '',
                                                                         '', '', 1,
                                                                         dropout_value)

                                cross_validation_users(c, model, x_array, y_array, model_name)

                                # model.save('tf_models/test_model_' + str(c))

                                c += 1
                    else:

                        model = dl_models.get_model_lstm(dense_input, 0, '', dense_output_activation, n_lstm_units,
                                                         loss_type, optimizer_type, 1, dropout_value)

                        experiments.utilities.write_line_results(filename_results, c, 'LSTM', label_name, input_name,
                                                                 loss_type, optimizer_type, dense_input, '', '',
                                                                 dense_output_activation, n_lstm_units, '', '', '',
                                                                 1, dropout_value)

                        cross_validation_users(c, model, x_array, y_array, model_name)

                        # model.save('tf_models/test_model_' + str(c))

                        c += 1

    return c


def iterate_cnn1d_lstm(c, x_array, y_array, loss_type, optimizer_type, label_name, input_name):
    model_name = 'CNN1D_LSTM'

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

                                            experiments.utilities.write_line_results(filename_results, c, model_name,
                                                                                     label_name,
                                                                                     input_name,
                                                                                     loss_type,
                                                                                     optimizer_type, dense_input,
                                                                                     dense_input_dim,
                                                                                     dense_input_activation,
                                                                                     dense_output_activation,
                                                                                     n_lstm_units,
                                                                                     n_cnn_filters,
                                                                                     cnn_kernel_size, cnn_pool_size,
                                                                                     1, dropout_value)

                                            cross_validation_users(c, model, x_array, y_array, model_name)

                                            # model.save('tf_models/test_model_' + str(c))

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

                                    experiments.utilities.write_line_results(filename_results, c, model_name,
                                                                             label_name,
                                                                             input_name,
                                                                             loss_type,
                                                                             optimizer_type, dense_input,
                                                                             '', '',
                                                                             dense_output_activation, n_lstm_units,
                                                                             n_cnn_filters,
                                                                             cnn_kernel_size, cnn_pool_size,
                                                                             1,
                                                                             dropout_value)

                                    cross_validation_users(c, model, x_array, y_array, model_name)

                                    # model.save('tf_models/test_model_' + str(c))

                                    c += 1
    return c


def iterate_cnn1d_lstm_3dense(c, x_array, y_array, loss_type, optimizer_type, label_name, input_name):
    model_name = 'CNN1D_LSTM_3xDENSE'

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

                                            experiments.utilities.write_line_results(filename_results,
                                                                                     c, model_name, label_name,
                                                                                     input_name,
                                                                                     loss_type,
                                                                                     optimizer_type, dense_input,
                                                                                     dense_input_dim,
                                                                                     dense_input_activation,
                                                                                     dense_output_activation,
                                                                                     n_lstm_units,
                                                                                     n_cnn_filters,
                                                                                     cnn_kernel_size, cnn_pool_size,
                                                                                     1, dropout_value)

                                            cross_validation_users(c, model, x_array, y_array, model_name)

                                            # model.save('tf_models/test_model_' + str(c))

                                            c += 1
                                else:

                                    model = dl_models.get_model_cnn1d_lstm_3x_dense(dense_input, 0, '',
                                                                                    dense_output_activation,
                                                                                    n_cnn_filters,
                                                                                    cnn_kernel_size,
                                                                                    cnn_pool_size,
                                                                                    n_lstm_units,
                                                                                    loss_type,
                                                                                    optimizer_type, 1,
                                                                                    dropout_value)

                                    experiments.utilities.write_line_results(filename_results,
                                                                             c, model_name, label_name,
                                                                             input_name,
                                                                             loss_type,
                                                                             optimizer_type, dense_input,
                                                                             '', '',
                                                                             dense_output_activation, n_lstm_units,
                                                                             n_cnn_filters,
                                                                             cnn_kernel_size, cnn_pool_size,
                                                                             1,
                                                                             dropout_value)

                                    cross_validation_users(c, model, x_array, y_array, model_name)

                                    # model.save('tf_models/test_model_' + str(c))

                                    c += 1
    return c


def iterate_2xcnn1d_lstm(c, x_array, y_array, loss_type, optimizer_type, label_name, input_name):
    model_name = '2xCNN1D_LSTM'

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

                                            experiments.utilities.write_line_results(filename_results, c, model_name,
                                                                                     label_name,
                                                                                     input_name,
                                                                                     loss_type,
                                                                                     optimizer_type, dense_input,
                                                                                     dense_input_dim,
                                                                                     dense_input_activation,
                                                                                     dense_output_activation,
                                                                                     n_lstm_units,
                                                                                     n_cnn_filters,
                                                                                     cnn_kernel_size, cnn_pool_size,
                                                                                     1, dropout_value)

                                            cross_validation_users(c, model, x_array, y_array, model_name)

                                            # model.save('tf_models/test_model_' + str(c))

                                            c += 1
                                else:

                                    model = dl_models.get_model_2x_cnn1d_lstm(dense_input, 0, '',
                                                                              dense_output_activation,
                                                                              n_cnn_filters,
                                                                              cnn_kernel_size,
                                                                              cnn_pool_size,
                                                                              n_lstm_units,
                                                                              loss_type,
                                                                              optimizer_type, 1,
                                                                              dropout_value)

                                    experiments.utilities.write_line_results(filename_results, c, model_name,
                                                                             label_name,
                                                                             input_name,
                                                                             loss_type,
                                                                             optimizer_type, dense_input,
                                                                             '', '',
                                                                             dense_output_activation, n_lstm_units,
                                                                             n_cnn_filters,
                                                                             cnn_kernel_size, cnn_pool_size,
                                                                             1,
                                                                             dropout_value)

                                    cross_validation_users(c, model, x_array, y_array, model_name)

                                    # model.save('tf_models/test_model_' + str(c))

                                    c += 1
    return c


def iterate_cnn2d(c, x_array, y_array, loss_type, optimizer_type, label_name, input_name):
    model_name = 'CNN2D'

    x_array = np.expand_dims(x_array, 2)
    y_array = np.expand_dims(y_array, 2)

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

                                        experiments.utilities.write_line_results(filename_results, c, model_name,
                                                                                 label_name, input_name,
                                                                                 loss_type,
                                                                                 optimizer_type, dense_input,
                                                                                 dense_input_dim,
                                                                                 dense_input_activation,
                                                                                 dense_output_activation, '',
                                                                                 n_cnn_filters,
                                                                                 cnn_kernel_size, cnn_pool_size, 1,
                                                                                 dropout_value)

                                        cross_validation_users(c, model, x_array, y_array, model_name)

                                        # model.save('tf_models/test_model_' + str(c))

                                        c += 1
                            else:

                                model = dl_models.get_model_cnn2d(dense_input, 0, '', dense_output_activation,
                                                                  n_cnn_filters, cnn_kernel_size, cnn_pool_size,
                                                                  loss_type, optimizer_type, 1, dropout_value)

                                experiments.utilities.write_line_results(filename_results, c, model_name, label_name,
                                                                         input_name, loss_type, optimizer_type,
                                                                         dense_input,
                                                                         '', '', dense_output_activation, '',
                                                                         n_cnn_filters, cnn_kernel_size,
                                                                         cnn_pool_size, 1, dropout_value)

                                cross_validation_users(c, model, x_array, y_array, model_name)

                                # model.save('tf_models/test_model_' + str(c))

                                c += 1

    return c


def iterate_cnn2d_lstm(c, x_array, y_array, loss_type, optimizer_type, label_name, input_name):
    model_name = 'CNN2D_LSTM'

    x_array = np.expand_dims(x_array, 2)
    y_array = np.expand_dims(y_array, 2)

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

                                            experiments.utilities.write_line_results(filename_results, c, model_name,
                                                                                     label_name,
                                                                                     input_name,
                                                                                     loss_type,
                                                                                     optimizer_type, dense_input,
                                                                                     dense_input_dim,
                                                                                     dense_input_activation,
                                                                                     dense_output_activation,
                                                                                     n_lstm_units,
                                                                                     n_cnn_filters,
                                                                                     cnn_kernel_size, cnn_pool_size, 1,
                                                                                     dropout_value)

                                            cross_validation_users(c, model, x_array, y_array, model_name)

                                            # model.save('tf_models/test_model_' + str(c))

                                            c += 1
                                else:

                                    model = dl_models.get_model_cnn2d_lstm(dense_input, 0, n_lstm_units,
                                                                           dense_output_activation,
                                                                           n_cnn_filters, cnn_kernel_size,
                                                                           cnn_pool_size,
                                                                           n_lstm_units, loss_type,
                                                                           optimizer_type, dropout_value)

                                    experiments.utilities.write_line_results(filename_results, c, 'CNN2D_LSTM',
                                                                             label_name, input_name,
                                                                             loss_type,
                                                                             optimizer_type, dense_input,
                                                                             '', '',
                                                                             dense_output_activation, n_lstm_units,
                                                                             n_cnn_filters,
                                                                             cnn_kernel_size, cnn_pool_size, 1,
                                                                             dropout_value)

                                    cross_validation_users(c, model, x_array, y_array, model_name)

                                    # model.save('tf_models/test_model_' + str(c))

                                    c += 1

    return c
