import numpy
from matplotlib import pyplot as plt
import pandas as pd

sync_dataframe = pd.read_csv('datasets/eeg/eeg_user_3.csv')

x = []
time = []

plt.plot(sync_dataframe[' time'][:200], sync_dataframe[' Delta'][:200], color='b', label='delta channel')
plt.savefig('test_plot.png')