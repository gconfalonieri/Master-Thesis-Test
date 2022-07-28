import os
import toml
import experiments
import models.plots

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras import Sequential
from keras.layers import LSTM, Dense, Conv1D, MaxPooling1D
from sklearn.model_selection import train_test_split
import numpy as np

config = toml.load('config.toml')
experiments.utilities.fix_seeds()

model = Sequential()
model.add(Conv1D(filters=256, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=4))
model.add(LSTM(32))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

complete_x_list = models.utilities.get_users_oversampled_array()
complete_y_list = models.utilities.get_labels_users_array()

i = 1

for user_x, user_y in zip(complete_x_list, complete_y_list):
    print('USER ' + str(i))
    X_train, X_test, y_train, y_test = train_test_split(user_x, user_y, test_size=0.2, shuffle=True)
    X_train = np.array(X_train).astype(np.float32)
    X_test = np.asarray(X_test).astype(np.float32)
    history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))
    results = model.evaluate(X_test, y_test)
    print(results)
    i += 1