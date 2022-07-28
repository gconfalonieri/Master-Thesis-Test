import numpy
import toml
import pandas as pd
import numpy as np
import scipy.signal
import sklearn.utils
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split

config = toml.load('config.toml')


def get_max_series_len():

    max_len = 0

    for i in range(1, 53):
        user_id = 'USER_' + str(i)
        if i not in config['general']['excluded_users']:
            path = config['path']['sync_prefix'] + 'sync_dataset_' + user_id.lower() + '.csv'
            df_sync = pd.read_csv(path)
            media_names = df_sync.drop_duplicates('media_name', keep='last')['media_name']
            for name in media_names:
                reduced_df = df_sync[df_sync['media_name'] == name]
                curr_len = len(reduced_df.iloc[:, 0])
                if curr_len > max_len:
                    max_len = curr_len

    return max_len


def get_max_series_len_shifted():

    max_len = 0

    for i in range(1, 53):
        user_id = 'USER_' + str(i)
        if i not in config['general']['excluded_users']:
            path = config['path']['sync_prefix'] + 'sync_dataset_' + user_id.lower() + '.csv'
            df_sync = pd.read_csv(path)
            media_names = df_sync.drop_duplicates('media_name', keep='last')['media_name']
            for name in media_names:
                reduced_df = df_sync[df_sync['media_name'] == name]
                for j in range(0, 14):
                    shifted = reduced_df.iloc[j::15, :]
                    curr_len = len(shifted.iloc[:, 0])
                    if curr_len > max_len:
                        max_len = curr_len
    return max_len


def get_min_series_len_shifted():

    min_len = float('inf')

    for i in range(1, 53):
        user_id = 'USER_' + str(i)
        if i not in config['general']['excluded_users']:
            path = config['path']['sync_prefix'] + 'sync_dataset_' + user_id.lower() + '.csv'
            df_sync = pd.read_csv(path)
            media_names = df_sync.drop_duplicates('media_name', keep='last')['media_name']
            for name in media_names:
                reduced_df = df_sync[df_sync['media_name'] == name]
                for j in range(0, 14):
                    shifted = reduced_df.iloc[j::15, :]
                    curr_len = len(shifted.iloc[:, 0])
                    if curr_len < min_len:
                        min_len = curr_len
    return min_len



def get_max_validation_len_shifted():

    max_len = 0

    for i in range(1, 53):
        user_id = 'USER_' + str(i)
        if i in config['general']['validation_users']:
            path = config['path']['sync_validation_prefix'] + 'sync_dataset_' + user_id.lower() + '.csv'
            df_sync = pd.read_csv(path)
            media_names = df_sync.drop_duplicates('media_name', keep='last')['media_name']
            for name in media_names:
                reduced_df = df_sync[df_sync['media_name'] == name]
                for j in range(0, 14):
                    shifted = reduced_df.iloc[j::15, :]
                    curr_len = len(shifted.iloc[:, 0])
                    if curr_len > max_len:
                        max_len = curr_len
    return max_len

def get_max_series_df():

    max_len = 0
    max_series_df = pd.DataFrame()

    for i in range(1, 53):
        user_id = 'USER_' + str(i)
        if i not in config['general']['excluded_users']:
            path = config['path']['sync_prefix'] + 'sync_dataset_' + user_id.lower() + '.csv'
            df_sync = pd.read_csv(path)
            media_names = df_sync.drop_duplicates('media_name', keep='last')['media_name']
            for name in media_names:
                reduced_df = df_sync[df_sync['media_name'] == name]
                curr_len = len(reduced_df['time'])
                if curr_len > max_len:
                    max_len = curr_len
                    max_series_df = reduced_df

    return max_series_df


def get_questions_array():

    complete_x_list = []

    for i in range(1, 53):
        user_id = 'USER_' + str(i)
        if i not in config['general']['excluded_users']:
            path = config['path']['sync_prefix']+ 'sync_dataset_' + user_id.lower() + '.csv'
            df_sync = pd.read_csv(path)
            media_names = df_sync.drop_duplicates('media_name', keep='last')['media_name']
            for name in media_names:
                question_list = []
                reduced_df = df_sync[df_sync['media_name'] == name]
                for f in config['algorithm']['gaze_features']:
                    arr = np.asarray(reduced_df[f]).astype('float32')
                    question_list.append(arr)
                complete_x_list.append(question_list)

    return np.array(complete_x_list, dtype=np.ndarray)


def get_questions_oversampled_array_shifted():

    complete_x_list = []

    max_len = get_max_series_len_shifted()
    # max_len = config['computed']['shifted_max_len']

    for i in range(1, 53):
        user_id = 'USER_' + str(i)
        if i not in config['general']['excluded_users']:
            path = config['path']['sync_prefix'] + 'sync_dataset_' + user_id.lower() + '.csv'
            df_sync = pd.read_csv(path)
            media_names = df_sync.drop_duplicates('media_name', keep='last')['media_name']
            for name in media_names:
                reduced_df = df_sync[df_sync['media_name'] == name]
                for j in range(0, 14):
                    shifted = reduced_df.iloc[j::15, :]
                    question_list = []
                    for f in config['algorithm']['gaze_features']:
                        arr = np.asarray(shifted[f]).astype('float32')
                        oversampled_array = numpy.array(0)
                        if config['preprocessing']['resample_library'] == 'sklearn':
                            oversampled_array = sklearn.utils.resample(arr, n_samples=max_len, stratify=arr)
                        elif config['preprocessing']['resample_library'] == 'scipy':
                            oversampled_array = scipy.signal.resample(arr, max_len)
                        question_list.append(oversampled_array)
                    complete_x_list.append(question_list)

    return np.array(complete_x_list, dtype=np.ndarray)


def get_questions_oversampled_validation_shifted():

    complete_x_list = []

    max_len = get_max_series_len_shifted()
    # max_len = config['computed']['shifted_max_len']

    for i in range(1, 53):
        user_id = 'USER_' + str(i)
        if i in config['general']['validation_users']:
            path = config['path']['sync_validation_prefix'] + 'sync_dataset_' + user_id.lower() + '.csv'
            df_sync = pd.read_csv(path)
            media_names = df_sync.drop_duplicates('media_name', keep='last')['media_name']
            for name in media_names:
                reduced_df = df_sync[df_sync['media_name'] == name]
                for j in range(0, 14):
                    shifted = reduced_df.iloc[j::15, :]
                    question_list = []
                    for f in config['algorithm']['gaze_features']:
                        arr = np.asarray(shifted[f]).astype('float32')
                        oversampled_array = numpy.array(0)
                        if config['preprocessing']['resample_library'] == 'sklearn':
                            oversampled_array = sklearn.utils.resample(arr, n_samples=max_len, stratify=arr)
                        elif config['preprocessing']['resample_library'] == 'scipy':
                            oversampled_array = scipy.signal.resample(arr, max_len)
                        question_list.append(oversampled_array)
                    complete_x_list.append(question_list)

    return np.array(complete_x_list, dtype=np.ndarray)


def get_questions_oversampled_array():

    complete_x_list = []

    max_len = get_max_series_len()

    for i in range(1, 53):
        user_id = 'USER_' + str(i)
        if i not in config['general']['excluded_users']:
            path = config['path']['sync_prefix'] + 'sync_dataset_' + user_id.lower() + '.csv'
            df_sync = pd.read_csv(path)
            media_names = df_sync.drop_duplicates('media_name', keep='last')['media_name']
            for name in media_names:
                question_list = []
                reduced_df = df_sync[df_sync['media_name'] == name]
                for f in config['algorithm']['gaze_features']:
                    arr = np.asarray(reduced_df[f]).astype('float32')
                    oversampled_array = numpy.array(0)
                    if config['preprocessing']['resample_library'] == 'sklearn':
                        oversampled_array = sklearn.utils.resample(arr, n_samples=max_len, stratify=arr)
                    elif config['preprocessing']['resample_library'] == 'scipy':
                        oversampled_array = scipy.signal.resample(arr, max_len)
                    question_list.append(oversampled_array)
                complete_x_list.append(question_list)

    return np.array(complete_x_list, dtype=np.ndarray)


def get_labels_questions_array():

    complete_y_list = []

    path_labelled_df = config['path']['labelled_dataset']
    df_labelled = pd.read_csv(path_labelled_df)

    for i in df_labelled.index:
        arr = np.array(df_labelled['LABEL'][i])
        complete_y_list.append(np.expand_dims(arr, axis=(0)))

    return np.asarray(complete_y_list).astype('int')


def get_labels_questions_array_shifted():

    complete_y_list = []

    path_labelled_df = config['path']['labelled_dataset']
    df_labelled = pd.read_csv(path_labelled_df)

    for i in df_labelled.index:
        user_id = df_labelled['USER_ID'][i]
        id = int((user_id.split('_'))[1])
        if id not in config['general']['excluded_users']:
            print(id)
            arr = np.array(df_labelled['LABEL'][i])
            for j in range(0,14):
                complete_y_list.append(np.expand_dims(arr, axis=(0)))

    return np.asarray(complete_y_list).astype('int')


def get_labels_questions_validation_shifted():

    complete_y_list = []

    path_labelled_df = config['path']['labelled_dataset']
    df_labelled = pd.read_csv(path_labelled_df)

    for i in df_labelled.index:
        user_id = df_labelled['USER_ID'][i]
        id = int((user_id.split('_'))[1])
        if id in config['general']['validation_users']:
            arr = np.array(df_labelled['LABEL'][i])
            for j in range(0,14):
                complete_y_list.append(np.expand_dims(arr, axis=(0)))

    return np.asarray(complete_y_list).astype('int')


def get_arrays_shuffled_shifted(test_size_value):

    complete_x = []
    complete_y = []

    max_len = get_max_series_len_shifted()

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


def get_arrays_shuffled_shifted_thr(test_size_value):

    complete_x = []
    complete_y = []

    max_len = get_max_series_len_shifted()

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
                    counter = 0
                    if len(shifted) > config['algorithm']['min_len_thr']:
                        for f in config['algorithm']['gaze_features']:
                            arr = np.asarray(shifted[f]).astype('float32')
                            oversampled_array = numpy.array(0)
                            if config['preprocessing']['resample_library'] == 'sklearn':
                                oversampled_array = sklearn.utils.resample(arr, n_samples=max_len, stratify=arr)
                            elif config['preprocessing']['resample_library'] == 'scipy':
                                oversampled_array = scipy.signal.resample(arr, max_len)
                            feature_list.append(oversampled_array)
                        question_list.append(feature_list)
                if len(question_list) > 0:
                    complete_x.append(question_list)
                    red_df = df_labelled.loc[(df_labelled['USER_ID'] == user_id) & (df_labelled['MEDIA_NAME'] == name)]
                    arr = np.array(red_df['LABEL'].values[0])
                    complete_y.append(np.expand_dims(arr, axis=(0)))

    array_x = np.array(complete_x, dtype=np.ndarray)
    array_y = np.asarray(complete_y).astype('int')

    train_array, validation_array, label_train, label_validation = train_test_split(array_x, array_y, test_size=test_size_value, shuffle=True)

    new_train = []
    new_validation = []
    new_label_train = []
    new_label_validation = []

    index = 0
    for x in train_array:
        for y in x:
            new_train.append(y)
            new_label_train.append(label_train[index])
        index += 1
    index = 0
    for x in validation_array:
        for y in x:
            new_validation.append(y)
            new_label_validation.append(label_validation[index])
        index += 1

    return np.array(new_train).astype(np.float32), np.asarray(new_validation).astype(np.float32), \
           np.asarray(new_label_train).astype('int'), np.asarray(new_label_validation).astype('int')


def get_users_array_shifted():

    complete_x = []

    max_len = get_max_series_len_shifted()

    for i in range(1, 53):
        user_id = 'USER_' + str(i)
        if i not in config['general']['excluded_users']:
            path = config['path']['sync_prefix'] + 'sync_dataset_' + user_id.lower() + '.csv'
            df_sync = pd.read_csv(path)
            media_names = df_sync.drop_duplicates('media_name', keep='last')['media_name']
            user_list = []
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
                user_list.append(question_list)
            complete_x.append(user_list)

    return np.array(complete_x, dtype=np.ndarray)
