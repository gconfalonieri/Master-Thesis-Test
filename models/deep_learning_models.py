import os
from keras import Sequential, regularizers, Input
from keras.regularizers import l1, l2, l1_l2
from keras.layers import Conv1D, MaxPooling1D, Dense, LSTM, BatchNormalization, Conv2D, MaxPooling2D, Dropout, \
    TimeDistributed, Flatten, InputLayer, Reshape

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_model_dense_cnn1d_lstm():
    model = Sequential()
    model.add(Dense(1, activation='relu'))
    model.add(Conv1D(filters=256, kernel_size=5, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=4, padding='same'))
    model.add(LSTM(32))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model

def get_model_ccn1d(complete_x_list):
    model = Sequential()
    model.add(InputLayer(input_shape=complete_x_list[0].shape))
    model.add(Conv1D(filters=256, kernel_size=5, padding='same', activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(MaxPooling1D(pool_size=4, padding='same'))
    model.add(Dense(3, activation='linear'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model


def get_model_lstm(complete_x_list):
    model = Sequential()
    model.add(InputLayer(input_shape=complete_x_list[0].shape))
    model.add(LSTM(32))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model


def get_model_cnn1d_lstm(complete_x_list):
    model = Sequential()
    model.add(InputLayer(input_shape=complete_x_list[0].shape))
    model.add(Conv1D(filters=256, kernel_size=5, padding='same', activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(MaxPooling1D(pool_size=4, padding='same'))
    model.add(LSTM(32))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model


def get_model_2x_cnn1d_lstm(complete_x_list):
    model = Sequential()
    model.add(InputLayer(input_shape=complete_x_list[0].shape))
    model.add(Conv1D(filters=256, kernel_size=5, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=256, kernel_size=5, padding='same', activation='relu'))
    model.add(LSTM(32))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model


def get_model_cnn1d_lstm_3x_dense(complete_x_list):
    model = Sequential()
    model.add(InputLayer(input_shape=complete_x_list[0].shape))
    model.add(Conv1D(filters=256, kernel_size=5, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=4, padding='same'))
    model.add(LSTM(32))
    model.add(Dense(16))
    model.add(Dense(8))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model


def get_model_cnn2d(complete_x_list):
    model = Sequential()
    model.add(InputLayer(input_shape=complete_x_list[0].shape))
    model.add(Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=3, padding='same'))
    model.add(Flatten())
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model


def get_model_cnn2d_lstm(complete_x_list):
    model = Sequential()
    model.add(InputLayer(input_shape=complete_x_list[0].shape))
    model.add(Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=3, padding='same'))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(32))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model

