import os

import models.utilities

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.utils.np_utils import to_categorical
import tensorflow as tf
import keras.preprocessing
import pandas as pd
from keras import Sequential, Model
from keras.layers import LSTM, Dense, Input, Embedding, Conv1D, Conv2D, MaxPooling1D
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
X_train, X_test, y_train, y_test = train_test_split(complete_x_list, complete_y_list, test_size=0.2)

X_train = np.asarray(X_train).astype(np.float32)
X_test = np.asarray(X_test).astype(np.float32)

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

model = Sequential()
model.add(Conv1D(filters=256, kernel_size=2))
model.add(MaxPooling1D(pool_size=2, padding='same'))
# model.add(LSTM(32))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=200)


# model = Model(input_tensor, output_tensor, name='Model')
# model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
# model.fit(X_train, y_train, epochs=30, batch_size=32)
# print(model.summary())

# model.add(Input(shape=(len(users), config['general']['n_questions'], 1, )))
# model.add(Input())
# model.add(LSTM(32))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
# model.fit(X_train, y_train, epochs=30, batch_size=32)
# print(model.summary())
