import numpy
import toml
from numpy import load
from sklearn.model_selection import train_test_split

import models.utilities
import numpy as np
import pandas as pd

config = toml.load('config.toml')
path_labelled_df = config['path']['labelled_dataset']

# save numpy array as npy file
# all_windowed_array_data = load('all_windowed_array_data.npy', allow_pickle=True)
# all_windowed_array_labels = load('all_windowed_array_labels.npy')

complete_x_list = models.utilities.get_user_windowed_array(2)[0]
complete_y_list = models.utilities.get_user_windowed_array(2)[1]

print("# TRAIN SERIES #")

print(complete_x_list.shape)
print(type(complete_x_list))
print(complete_x_list[0].shape)
print(type(complete_x_list[0]))
print(complete_x_list[0][0].shape)
print(type(complete_x_list[0][0]))
print(type(complete_x_list[0][0][0]))

print("# TRAIN LABELS #")
print(complete_y_list.shape)
print(type(complete_y_list))
print(complete_y_list[0].shape)
print(type(complete_y_list[0]))
