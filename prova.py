import numpy
from matplotlib import pyplot as plt
import pandas as pd
import models.utilities

# sync_dataframe = pd.read_csv('datasets/eeg/eeg_user_3.csv')

# x = []
# time = []
# plt.plot(sync_dataframe[' time'][:200], sync_dataframe[' Delta'][:200], color='b', label='delta channel')
# plt.savefig('test_plot.png')

numpy_array = models.utilities.get_questions_oversampled_array()

data_arr = models.utilities.split_mask_array(numpy_array)
init_arr = data_arr[0]
mask_arr = data_arr[1]

print(init_arr.shape)
print(mask_arr.shape)