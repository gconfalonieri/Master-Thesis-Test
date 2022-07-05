import numpy as np
import pandas as pd
import toml

from experiments.utilities import get_input_array_string, get_labels_array_string

config = toml.load('config.toml')

results_dataframe = pd.DataFrame(columns=['index', 'label', 'input', 'loss_type', 'optimizer_type'])
c = 0
results_dataframe.to_csv('results.csv', index=False)

for label_array in config['path']['labels_arrays']:
    complete_y_list = np.load(label_array, allow_pickle=True)
    label_name = get_labels_array_string(label_array)
    for input_array in config['path']['input_arrays']:
        pd.read_csv('results.csv')
        complete_x_list = np.load(input_array, allow_pickle=True)
        input_name = get_input_array_string(input_array)
        for loss_type in config['algorithm']['loss_types']:
            for optimizer_type in config['algorithm']['optimizer_types']:
                results_dataframe.loc[c] = [c, label_name, input_name, loss_type, optimizer_type]
                results_dataframe.to_csv('results.csv', index=False)
                c += 1