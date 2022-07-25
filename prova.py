import random

import numpy
import toml
import pandas as pd
import numpy as np
import scipy.signal
import sklearn.utils
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split

import models

config = toml.load('config.toml')


def get_questions_oversampled_validation_shifted(test_size_value):

    complete_x = []
    complete_y = []

    max_len = models.utilities.get_max_series_len_shifted()

    path_labelled_df = config['path']['labelled_dataset']
    df_labelled = pd.read_csv(path_labelled_df)

    for i in range(1, 53):
        user_id = 'USER_' + str(i)
        if i not in config['general']['excluded_users']:
            path = config['path']['sync_prefix'] + 'sync_dataset_' + user_id.lower() + '.csv'
            df_sync = pd.read_csv(path)
            media_names = df_sync.drop_duplicates('media_name', keep='last')['media_name']
            for name in media_names:
                question_list = []
                reduced_df = df_sync[df_sync['media_name'] == name]
                for j in range(0, 14):
                    shifted = reduced_df.iloc[j::15, :]
                    feature_list = []
                    for f in config['algorithm']['gaze_features']:
                        arr = np.asarray(shifted[f]).astype('float32')
                        oversampled_array = numpy.array(0)
                        if config['preprocessing']['resample_library'] == 'sklearn':
                            oversampled_array = sklearn.utils.resample(arr, n_samples=max_len, stratify=arr)
                        elif config['preprocessing']['resample_library'] == 'scipy':
                            oversampled_array = scipy.signal.resample(arr, max_len)
                        feature_list.append(oversampled_array)
                    question_list.append(feature_list)
                red_df = df_labelled.loc[(df_labelled['USER_ID'] == user_id) & (df_labelled['MEDIA_NAME'] == name)]
                arr = np.array(red_df['LABEL'].values[0])
                complete_y.append(np.expand_dims(arr, axis=(0)))
                complete_x.append(question_list)

    array_x = np.array(complete_x, dtype=np.ndarray)
    array_y = np.asarray(complete_y).astype('int')

    train_array, validation_array, train_labels, validation_labels = train_test_split(array_x, array_y, test_size=test_size_value, shuffle=True)

    new_train = []
    new_validation = []
    new_label_train = []
    new_label_validation = []

    for x in train_array:
        for y in x:
            new_train.append(y)
    for x in validation_array:
        for y in x:
            new_validation.append(y)
    for x in train_labels:
        for i in range(0, 14):
            new_label_train.append(x)
    for x in validation_labels:
        for i in range(0, 14):
            new_label_validation.append(x)

    return np.array(new_train).astype(np.float32), np.asarray(new_validation).astype(np.float32), \
           np.asarray(new_label_train).astype('int'), np.asarray(new_label_validation).astype('int')


total_arr = get_questions_oversampled_validation_shifted(0.12)
X_train = total_arr[0]
X_test = total_arr[1]
y_train = total_arr[2]
y_test = total_arr[3]

print(X_train.shape)
print(type(X_train))

model = models.deep_learning_models.get_model_lstm(0, 0, '', 'relu', 32, 'mse', 'adam', 1, 0.1)
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), shuffle=True)