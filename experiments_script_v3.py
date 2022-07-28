import numpy as np
import pandas as pd
import toml
from sklearn.model_selection import train_test_split

import experiments.models_iterations
import models.deep_learning_models as dl_models

import models.deep_learning_models
from experiments.utilities import get_input_array_string, get_labels_array_string
import csv

from models.utilities import get_users_arrays_shifted

config = toml.load('config.toml')

c = 0

for label_array in config['path']['labels_arrays']:

    label_name = get_labels_array_string(label_array)

    array_total = get_users_arrays_shifted()

    x_array = array_total[0]
    y_array = array_total[1]

    for input_array in config['path']['input_arrays']:

        input_name = get_input_array_string(input_array)

        for loss_type in config['algorithm']['loss_types']:

            for optimizer_type in config['algorithm']['optimizer_types']:

                c = experiments.models_iterations.iterate_cnn1d(c, x_array, y_array, loss_type, optimizer_type, label_name, input_name)
                c = experiments.models_iterations.iterate_cnn1d_lstm_3dense(c, x_array, y_array, loss_type, optimizer_type, label_name, input_name)
                c = experiments.models_iterations.iterate_2xcnn1d_lstm(c, x_array, y_array, loss_type, optimizer_type, label_name, input_name)
                c = experiments.models_iterations.iterate_lstm(c, x_array, y_array, loss_type, optimizer_type, label_name, input_name)
                c = experiments.models_iterations.iterate_cnn1d(c, x_array, y_array, loss_type, optimizer_type, label_name, input_name)
                c = experiments.models_iterations.iterate_cnn2d_lstm(c, x_array, y_array, loss_type, optimizer_type, label_name, input_name)
                c = experiments.models_iterations.iterate_cnn2d(c, x_array, y_array, loss_type, optimizer_type, label_name, input_name)