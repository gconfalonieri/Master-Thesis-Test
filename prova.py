import numpy as np
import toml
from sklearn.model_selection import train_test_split

import models
from models.utilities import get_questions_arrays_shifted_validation_thr, get_users_arrays_shifted_thr, \
    get_questions_arrays_shifted_thr, get_users_arrays_shifted, get_questions_arrays_shifted, \
    get_questions_arrays_shifted_validation

config = toml.load('config.toml')

total_arr = get_questions_arrays_shifted_validation(is_ordered=True)

print(total_arr[0].shape)
print(total_arr[1].shape)

X_train = models.utilities.aggregate_questions(total_arr[0], True)
y_train = models.utilities.aggregate_questions_labels(total_arr[1], False)

# X_train, X_test, y_train, y_test = train_test_split(total_arr[0], total_arr[1], test_size=0.2, shuffle=True)

# X_train = models.utilities.aggregate_questions(X_train, False)
# X_test = models.utilities.aggregate_questions(X_test, True)
# y_train = models.utilities.aggregate_questions_labels(y_train, False)
# y_test = models.utilities.aggregate_questions_labels(y_test, True)

print(X_train.shape)
# print(X_test.shape)
print(y_train.shape)
# print(y_test.shape)

# X_train = np.array(X_train).astype(np.float32)
# X_test = np.asarray(X_test).astype(np.float32)

# model = models.deep_learning_models.get_model_cnn1d_lstm(1, models.utilities.get_max_series_len(), 'relu', 'relu', 8, 4, 2, 64,  'mse', 'adam', 1, 0.2)
# history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), shuffle=True)
