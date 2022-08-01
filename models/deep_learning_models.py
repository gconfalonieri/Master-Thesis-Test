import toml
from keras import Sequential
from keras.regularizers import l2
from keras.layers import Conv1D, MaxPooling1D, Dense, LSTM, BatchNormalization, Conv2D, MaxPooling2D, Dropout, \
    TimeDistributed, Flatten
import experiments

config = toml.load('config.toml')
experiments.utilities.fix_seeds()


def get_model_ccn1d(dense_input, dense_input_dim, dense_input_activation, dense_output_activation,
                    n_cnn_filters, cnn_kernel_size, cnn_pool_size, loss_type, optimizer_type,
                    dropout, dropout_value):
    model = Sequential()
    if dense_input:
        model.add(Dense(dense_input_dim, activation=dense_input_activation))
    model.add(Conv1D(filters=n_cnn_filters, kernel_size=cnn_kernel_size, padding='same', activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(MaxPooling1D(pool_size=cnn_pool_size, padding='same'))
    if dropout:
        model.add(Dropout(dropout_value))
    model.add(Dense(1, dense_output_activation))
    model.compile(loss=loss_type, optimizer=optimizer_type, metrics=['accuracy'])
    return model


def get_model_lstm(dense_input, dense_input_dim, dense_input_activation, dense_output_activation, n_lstm_units,
                   loss_type, optimizer_type, dropout, dropout_value):
    model = Sequential()
    if dense_input:
        model.add(Dense(dense_input_dim, activation=dense_input_activation))
    model.add(LSTM(n_lstm_units))
    if dropout:
        model.add(Dropout(dropout_value))
    model.add(Dense(1, activation=dense_output_activation))
    model.compile(loss=loss_type, optimizer=optimizer_type, metrics=['accuracy'])
    return model


def get_model_cnn1d_lstm(dense_input, dense_input_dim,
                         dense_input_activation, dense_output_activation, n_cnn_filters, cnn_kernel_size,
                         cnn_pool_size, n_lstm_units, loss_type, optimizer_type, dropout, dropout_value):
    model = Sequential()
    if dense_input:
        model.add(Dense(dense_input_dim, activation=dense_input_activation))
    model.add(Conv1D(filters=n_cnn_filters, kernel_size=cnn_kernel_size, padding='same', activation='relu',
                     kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(MaxPooling1D(pool_size=cnn_pool_size, padding='same'))
    if dropout:
        model.add(Dropout(dropout_value))
    model.add(LSTM(n_lstm_units))
    model.add(Dense(1, activation=dense_output_activation))
    model.compile(loss=loss_type, optimizer=optimizer_type, metrics=['accuracy'])
    return model


def get_model_2x_cnn1d_lstm(dense_input, dense_input_dim,
                         dense_input_activation, dense_output_activation, n_cnn_filters, cnn_kernel_size,
                         cnn_pool_size, n_lstm_units, loss_type, optimizer_type, dropout, dropout_value):
    model = Sequential()
    if dense_input:
        model.add(Dense(dense_input_dim, activation=dense_input_activation))
    model.add(Conv1D(filters=n_cnn_filters, kernel_size=cnn_kernel_size, padding='same', activation='relu',
                     kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(MaxPooling1D(pool_size=cnn_pool_size))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=256, kernel_size=5, padding='same', activation='relu',
                     kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(LSTM(n_lstm_units))
    if dropout:
        model.add(Dropout(dropout_value))
    model.add(Dense(1, activation=dense_output_activation))
    model.compile(loss=loss_type, optimizer=optimizer_type, metrics=['accuracy'])
    return model


def get_model_cnn1d_lstm_3x_dense(dense_input, dense_input_dim,
                         dense_input_activation, dense_output_activation, n_cnn_filters, cnn_kernel_size,
                         cnn_pool_size, n_lstm_units, loss_type, optimizer_type, dropout, dropout_value):
    model = Sequential()
    if dense_input:
        model.add(Dense(dense_input_dim, activation=dense_input_activation))
    model.add(Conv1D(filters=n_cnn_filters, kernel_size=cnn_kernel_size, padding='same', activation='relu',
                     kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(MaxPooling1D(pool_size=cnn_pool_size, padding='same'))
    model.add(LSTM(n_lstm_units))
    if dropout:
        model.add(Dropout(dropout_value))
    model.add(Dense(32))
    model.add(Dense(16))
    model.add(Dense(8))
    model.add(Dense(1, activation=dense_output_activation))
    model.compile(loss=loss_type, optimizer=optimizer_type, metrics=['accuracy'])
    return model


def get_model_cnn2d(dense_input, dense_input_dim, dense_input_activation, dense_output_activation,
                    n_cnn_filters, cnn_kernel_size, cnn_pool_size, loss_type, optimizer_type, dropout, dropout_value):
    model = Sequential()
    if dense_input:
        model.add(Dense(dense_input_dim, activation=dense_input_activation))
    model.add(Conv2D(filters=n_cnn_filters, kernel_size=cnn_kernel_size, padding='same', activation='relu',
                     kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(MaxPooling2D(pool_size=cnn_pool_size, padding='same'))
    if dropout:
        model.add(Dropout(dropout_value))
    model.add(Flatten())
    model.add(Dense(1, activation=dense_output_activation))
    model.compile(loss=loss_type, optimizer=optimizer_type, metrics=['accuracy'])
    return model


def get_model_cnn2d_lstm(dense_input, dense_input_dim, dense_input_activation, dense_output_activation,
                         n_cnn_filters, cnn_kernel_size, cnn_pool_size, n_lstm_units, loss_type, optimizer_type, dropout_value):
    model = Sequential()
    if dense_input:
        model.add(Dense(dense_input_dim, activation=dense_input_activation))
    model.add(Conv2D(filters=n_cnn_filters, kernel_size=cnn_kernel_size, padding='same', activation='relu',
                     kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(MaxPooling2D(pool_size=cnn_pool_size, padding='same'))
    model.add(Dropout(dropout_value))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(n_lstm_units))
    model.add(Dense(1, activation=dense_output_activation))
    model.compile(loss=loss_type, optimizer=optimizer_type, metrics=['accuracy'])
    return model

