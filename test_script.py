import os

from matplotlib import pyplot as plt

import models.utilities
import models.plots

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.utils.np_utils import to_categorical
import tensorflow as tf
import keras.preprocessing
import pandas as pd
from keras import Sequential, Model
from keras.layers import LSTM, Dense, Input, Embedding, Conv1D, Conv2D, MaxPooling1D, Flatten
from sklearn.model_selection import train_test_split
import numpy as np
import toml

complete_x_list = models.utilities.get_questions_padded_array()
complete_y_list = models.utilities.get_labels_questions_array()


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

# X_train, X_test, y_train, y_test = train_test_split(np.ones((46 * 24, 2, 100)), np.ones((46 * 24, 1)), test_size=0.2)
X_train, X_test, y_train, y_test = train_test_split(complete_x_list, complete_y_list, test_size=0.2, shuffle=True)

X_train = np.array(X_train).astype(np.float32)
X_test = np.asarray(X_test).astype(np.float32)

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

model = Sequential()
model.add(Conv1D(filters=256, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=4))
model.add(LSTM(64))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
history_dict = history.history

models.plots.plot_model_loss(history_dict)
plt.clf()
models.plots.plot_model_accuracy(history_dict)

results = model.evaluate(X_test, y_test)
print(results)

# print(model.summary())