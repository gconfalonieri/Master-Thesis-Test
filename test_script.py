import os

from keras.utils.vis_utils import plot_model
from matplotlib import pyplot as plt
from numpy import load

import models.utilities
import models.plots

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.utils.np_utils import to_categorical
import tensorflow as tf
import keras.preprocessing
import pandas as pd
from keras import Sequential, Model
from keras.layers import LSTM, Dense, Input, Embedding, Conv1D, Conv2D, MaxPooling1D, Flatten, MaxPooling2D, \
    TimeDistributed, ReLU, BatchNormalization, GlobalAveragePooling1D, Dropout, ConvLSTM2D
from sklearn.model_selection import train_test_split
import numpy as np
import toml

loss_list = []
accuracy_list = []

def get_model_1():
    model = Sequential()
    model.add(Conv1D(filters=256, kernel_size=5, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(LSTM(32))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model

def make_model():

    model = Sequential()
    model.add(Conv1D(filters=256, kernel_size=5, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=256, kernel_size=5, padding='same', activation='relu'))
    model.add(LSTM(32))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    return model

complete_x_list = models.utilities.get_questions_oversampled_array()
complete_y_list = models.utilities.get_labels_questions_array()

print("# TRAIN SERIES #")
print(complete_x_list.shape)

print("# TRAIN LABELS #")
print(complete_y_list.shape)


# X_train, X_test, y_train, y_test = train_test_split(np.ones((46 * 24, 2, 100)), np.ones((46 * 24, 1)), test_size=0.2)
X_train, X_test, y_train, y_test = train_test_split(complete_x_list, complete_y_list, test_size=0.2, shuffle=True)

X_train = np.array(X_train).astype(np.float32)
X_test = np.asarray(X_test).astype(np.float32)

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

model = make_model()

history = model.fit(X_train, y_train, epochs=500, validation_data=(X_test, y_test))

history_dict = history.history

models.plots.plot_model_loss(history_dict)
plt.clf()
models.plots.plot_model_accuracy(history_dict)

results = model.evaluate(X_test, y_test)

print(results)

#loss_list.append(results[0])
#accuracy_list.append(results[1])

#df_results = pd.DataFrame(columns=['loss', 'accuracy'])

#df_results['loss'] = results[0]
#df_results['accuracy'] = results[1]
#df_results.to_csv('model_results/questions_oversampled.csv', index=False)

# print(model.summary())