import os

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import models.utilities
import models.deep_learning_models
from sklearn.model_selection import train_test_split
import numpy as np

complete_x_list = models.utilities.get_questions_oversampled_array()
np.save('datasets/arrays/undersampled_shifted/input_1_1_oversampled.npy', complete_x_list)
# complete_x_list = np.load('datasets/arrays/undersample_shifted/input_1_1_oversampled.npy', allow_pickle=True)
# print(complete_x_list.shape)