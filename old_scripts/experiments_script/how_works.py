import numpy as np
import toml
from sklearn.model_selection import train_test_split

import models.utilities
from models.utilities import get_questions_arrays_shifted_thr
from old_scripts.old_functions import get_questions_array_shifted_validation

config = toml.load('config.toml')

total_arr = ()


X_train, X_test, y_train, y_test = train_test_split(total_arr[0], total_arr[1], test_size=0.2, shuffle=True)

X_train = models.utilities.aggregate_questions(X_train)
X_test = models.utilities.aggregate_questions(X_test)
y_train = models.utilities.aggregate_questions_labels(y_train)
y_test = models.utilities.aggregate_questions_labels(y_test)

X_train = np.array(X_train).astype(np.float32)
X_test = np.asarray(X_test).astype(np.float32)

model = models.deep_learning_models.get_model_cnn1d_lstm(1, models.utilities.get_max_series_len(), 'relu', 'relu', 8, 4, 2, 64,  'mse', 'adam', 1, 0.2)
history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), shuffle=True)