import numpy as np
import pandas as pd
import toml
import models.deep_learning_models as dl_models

import models.deep_learning_models
from experiments.utilities import get_input_array_string, get_labels_array_string

config = toml.load('config.toml')

results_dataframe = pd.DataFrame(columns=['index', 'model', 'label', 'input', 'loss_type', 'optimizer_type',
                                          'dense_input', 'dense_input_activation', 'dense_input_activation',
                                          'lstm_units', 'cnn_fiters', 'cnn_kernel', 'dropout', 'dropout_value'])
c = 0
results_dataframe.to_csv('results.csv', index=False)

for label_array in config['path']['labels_arrays']:
    complete_y_list = np.load(label_array, allow_pickle=True)
    label_name = get_labels_array_string(label_array)
    for input_array in config['path']['input_arrays']:
        complete_x_list = np.load(input_array, allow_pickle=True)
        input_name = get_input_array_string(input_array)
        for loss_type in config['algorithm']['loss_types']:
            for optimizer_type in config['algorithm']['optimizer_types']:

                # CNN 1D
                pd.read_csv('results.csv')
                for dense_input in config['algorithm']['dense_input']:
                    for dense_input_activation in config['algorithm']['activation_types']:
                        for dense_output_activation in config['algorithm']['activation_types']:
                            for n_cnn_filters in config['algorithm']['n_cnn_filters']:
                                for cnn_kernel_size in config['algorithm']['cnn_kernel_size']:
                                    for dropout_value in config['algorithm']['dropout_value']:

                                        results_dataframe.loc[c] = [c, 'CNN1D', label_name, input_name,
                                                                            loss_type,
                                                                            optimizer_type, dense_input,
                                                                            dense_input_activation,
                                                                            dense_output_activation, '', n_cnn_filters,
                                                                            cnn_kernel_size, 1, dropout_value]
                                        c += 1
                results_dataframe.to_csv('results.csv', index=False)

                pd.read_csv('results.csv')
                # LSTM
                for dense_input in config['algorithm']['dense_input']:
                    for dense_input_activation in config['algorithm']['activation_types']:
                        for dense_output_activation in config['algorithm']['activation_types']:
                            for n_lstm_units in config['algorithm']['n_lstm_units']:
                                for dropout_value in config['algorithm']['dropout_value']:

                                    model = dl_models.get_model_lstm(dense_input, dense_input_activation, dense_output_activation,
                                                                 n_lstm_units, loss_type, optimizer_type, 1, dropout_value)
                                    results_dataframe.loc[c] = [c, 'LSTM', label_name, input_name, loss_type,
                                                            optimizer_type, dense_input, dense_input_activation,
                                                            dense_output_activation, n_lstm_units, '', '', 1, dropout_value]
                                    c += 1

                results_dataframe.to_csv('results.csv', index=False)