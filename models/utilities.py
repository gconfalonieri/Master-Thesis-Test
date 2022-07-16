import numpy
import toml
import pandas as pd
import numpy as np
import scipy.signal
import sklearn.utils
from scipy.interpolate import interp1d

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
                for j in range(0, 14):
                    shifted = reduced_df.iloc[j::15, :]
                    curr_len = len(shifted.iloc[:, 0])
                    if curr_len > max_len:
                        max_len = curr_len
    return max_len


def get_min_series_len():

    min_len = float('inf')

    for i in range(1, 53):
        user_id = 'USER_' + str(i)
        if i not in config['general']['excluded_users']:
            path = config['path']['sync_prefix'] + 'sync_dataset_' + user_id.lower() + '.csv'
            df_sync = pd.read_csv(path)
            media_names = df_sync.drop_duplicates('media_name', keep='last')['media_name']
            for name in media_names:
                reduced_df = df_sync[df_sync['media_name'] == name]
                curr_len = len(reduced_df['time'])
                if curr_len < min_len:
                    min_len = curr_len

    return min_len


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


def get_questions_padded_array():

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
                    pad_len = max_len - len(arr)
                    padded_array = np.pad(arr, pad_width=(pad_len, 0), mode='constant', constant_values=0)
                    # padded_array = np.pad(arr, pad_width=(0, pad_len), mode='constant', constant_values=0)
                    question_list.append(padded_array)
                complete_x_list.append(question_list)

    return np.array(complete_x_list, dtype=np.ndarray)


def get_questions_oversampled_array():

    complete_x_list = []

    max_len = config['computed']['shifted_max_len']

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


def get_fpogv_mask_array():

    complete_x_list = []

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
                    for x in shifted['FPOGV']:
                        question_list.append(x)
                    complete_x_list.append(np.array(question_list, dtype=np.ndarray))

    return np.array(complete_x_list, dtype=np.ndarray)


def get_labels_questions_array():

    complete_y_list = []

    path_labelled_df = config['path']['labelled_dataset']
    df_labelled = pd.read_csv(path_labelled_df)

    for i in df_labelled.index:
        arr = np.array(df_labelled['LABEL'][i])
        for j in range(0,14):
            complete_y_list.append(np.expand_dims(arr, axis=(0)))

    return np.asarray(complete_y_list).astype('int')


