import numpy
import toml
import pandas as pd
import numpy as np
import scipy.signal
import sklearn.utils

config = toml.load('config.toml')


def get_max_series_len():

    max_len = 0

    for i in range(1, 53):
        user_id = 'USER_' + str(i)
        if i not in config['general']['excluded_users']:
            path = 'datasets/sync_datasets/sync_dataset_' + user_id.lower() + '.csv'
            df_sync = pd.read_csv(path)
            media_names = df_sync.drop_duplicates('media_name', keep='last')['media_name']
            for name in media_names:
                reduced_df = df_sync[df_sync['media_name'] == name]
                curr_len = len(reduced_df['time'])
                if curr_len > max_len:
                    max_len = curr_len

    return max_len


def get_max_series_df():

    max_len = 0
    max_series_df = pd.DataFrame()

    for i in range(1, 53):
        user_id = 'USER_' + str(i)
        if i not in config['general']['excluded_users']:
            path = 'datasets/sync_datasets/sync_dataset_' + user_id.lower() + '.csv'
            df_sync = pd.read_csv(path)
            media_names = df_sync.drop_duplicates('media_name', keep='last')['media_name']
            for name in media_names:
                reduced_df = df_sync[df_sync['media_name'] == name]
                curr_len = len(reduced_df['time'])
                if curr_len > max_len:
                    max_len = curr_len
                    max_series_df = reduced_df

    return max_series_df


def get_users_array():

    complete_x_list = []

    for i in range(1, 53):
        user_id = 'USER_' + str(i)
        if i not in config['general']['excluded_users']:
            user_list = []
            path = 'datasets/sync_datasets/sync_dataset_' + user_id.lower() + '.csv'
            df_sync = pd.read_csv(path)
            media_names = df_sync.drop_duplicates('media_name', keep='last')['media_name']
            for name in media_names:
                reduced_df = df_sync[df_sync['media_name'] == name]
                question_list = []
                for f in config['algorithm']['eeg_features']:
                    question_list.append(np.asarray(reduced_df[f]))
                user_list.append(question_list)
            complete_x_list.append(user_list)

    return np.array(complete_x_list, dtype=np.ndarray)


def get_users_padded_array():

    complete_x_list = []
    max_len = get_max_series_len()

    for i in range(1, 53):
        user_id = 'USER_' + str(i)
        if i not in config['general']['excluded_users']:
            user_list = []
            path = 'datasets/sync_datasets/sync_dataset_' + user_id.lower() + '.csv'
            df_sync = pd.read_csv(path)
            media_names = df_sync.drop_duplicates('media_name', keep='last')['media_name']
            for name in media_names:
                reduced_df = df_sync[df_sync['media_name'] == name]
                question_list = []
                for f in config['algorithm']['eeg_features']:
                    arr = np.asarray(reduced_df[f]).astype('float32')
                    pad_len = max_len - len(arr)
                    padded_array = np.pad(arr, pad_width=(pad_len, 0), mode='mean')
                    question_list.append(padded_array)
                user_list.append(question_list)
            complete_x_list.append(user_list)

    return np.array(complete_x_list, dtype=np.ndarray)


def get_questions_array():

    complete_x_list = []

    for i in range(1, 53):
        user_id = 'USER_' + str(i)
        if i not in config['general']['excluded_users']:
            path = 'datasets/sync_datasets/sync_dataset_' + user_id.lower() + '.csv'
            df_sync = pd.read_csv(path)
            media_names = df_sync.drop_duplicates('media_name', keep='last')['media_name']
            for name in media_names:
                question_list = []
                reduced_df = df_sync[df_sync['media_name'] == name]
                for f in config['algorithm']['eeg_features']:
                    question_list.append(np.asarray(reduced_df[f]).astype('float32'))
                complete_x_list.append(question_list)

    return np.array(complete_x_list, dtype=np.ndarray)


def get_questions_padded_array():

    complete_x_list = []

    max_len = get_max_series_len()

    for i in range(1, 53):
        user_id = 'USER_' + str(i)
        if i not in config['general']['excluded_users']:
            path = 'datasets/sync_datasets/sync_dataset_' + user_id.lower() + '.csv'
            df_sync = pd.read_csv(path)
            media_names = df_sync.drop_duplicates('media_name', keep='last')['media_name']
            for name in media_names:
                question_list = []
                reduced_df = df_sync[df_sync['media_name'] == name]
                for f in config['algorithm']['eeg_features']:
                    arr = np.asarray(reduced_df[f]).astype('float32')
                    pad_len = max_len - len(arr)
                    padded_array = np.pad(arr, pad_width=(pad_len, 0), mode='constant', constant_values=0)
                    question_list.append(padded_array)
                complete_x_list.append(question_list)

    return np.array(complete_x_list, dtype=np.ndarray)


def get_questions_oversampled_array():

    complete_x_list = []

    max_len = get_max_series_len()

    for i in range(1, 53):
        user_id = 'USER_' + str(i)
        if i not in config['general']['excluded_users']:
            path = 'datasets/sync_datasets/sync_dataset_' + user_id.lower() + '.csv'
            df_sync = pd.read_csv(path)
            media_names = df_sync.drop_duplicates('media_name', keep='last')['media_name']
            for name in media_names:
                question_list = []
                reduced_df = df_sync[df_sync['media_name'] == name]
                for f in config['algorithm']['eeg_features']:
                    arr = np.asarray(reduced_df[f]).astype('float32')
                    oversampled_array = numpy.array()
                    if config['preprocessing']['resample_library'] == 'sklearn':
                        oversampled_array = sklearn.utils.resample(arr, n_samples=max_len, stratify=arr)
                    elif config['preprocessing']['resample_library'] == 'scikit':
                        oversampled_array = sklearn.utils.resample(arr, max_len)
                    question_list.append(oversampled_array)
                complete_x_list.append(question_list)

    return np.array(complete_x_list, dtype=np.ndarray)


def get_labels_users_array():

    c = 0
    complete_y_list = []

    path_labelled_df = config['path']['labelled_dataset']
    df_labelled = pd.read_csv(path_labelled_df)

    for i in range(1, 53):
        if i not in config['general']['excluded_users']:
            user_list = []
            for j in range(1, 25):
                user_list.append(df_labelled['LABEL'][c])
                c += 1
            complete_y_list.append(user_list)

    return np.asarray(complete_y_list).astype('int')


def get_labels_questions_array():

    c = 0
    complete_y_list = []

    path_labelled_df = config['path']['labelled_dataset']
    df_labelled = pd.read_csv(path_labelled_df)

    for i in df_labelled.index:
        arr = np.array(df_labelled['LABEL'][i])
        complete_y_list.append(np.expand_dims(arr, axis=(0)))

    return np.asarray(complete_y_list).astype('int')