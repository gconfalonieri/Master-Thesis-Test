import numpy
import numpy as np
import pandas as pd
import scipy
import sklearn
import toml
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split

from models.utilities import get_max_series_len_shifted

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
                print(curr_len)
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



def get_all_windowed_array():

    complete_x_list = []
    complete_y_list = []

    path_labelled_df = config['path']['labelled_dataset']
    df_labelled = pd.read_csv(path_labelled_df)

    min_len = get_min_series_len()
    win_len = min_len

    for i in range(1, 53):
        user_id = 'USER_' + str(i)
        if i not in config['general']['excluded_users']:
            path = config['path']['sync_prefix'] + 'sync_dataset_' + user_id.lower() + '.csv'
            df_sync = pd.read_csv(path)
            for f in config['algorithm']['eeg_features']:
                end_index = - 1
                max_index = int(len(df_sync['time']) / win_len)
                for i in range(0, max_index):
                    start_index = end_index + 1
                    end_index = start_index + win_len
                    arr = np.asarray(df_sync[f].iloc[start_index:end_index]).astype('float32')
                    if len(arr) == win_len:
                        complete_x_list.append(arr)
                        label_index = df_labelled.index[(df_labelled['MEDIA_NAME'] == df_sync['media_name'][start_index]) & (
                                    df_labelled['USER_ID'] == df_sync['user_id'][start_index])]
                        complete_y_list.append(df_labelled['LABEL'][label_index])

    return np.array(complete_x_list, dtype=np.ndarray), np.asarray(complete_y_list).astype('int')


def split_mask_array(numpy_array):

    data_array = []
    mask_array = []
    channels_arr = []
    bit_arr = []

    for questions in numpy_array:
        for channels in questions:
            channels_arr.append(channels[0])
            channels_arr.append(channels[1])
            channels_arr.append(channels[2])
            channels_arr.append(channels[3])
            bit_arr.append(channels[4])
        data_array.append(channels_arr)
        mask_array.append(bit_arr)

    return np.array(data_array, dtype=np.ndarray), np.asarray(mask_array).astype('int')


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

    return np.asarray(complete_x_list).astype(int)


def get_user_windowed_array(i):

    complete_x_list = []
    complete_y_list = []

    path_labelled_df = config['path']['labelled_dataset']
    df_labelled = pd.read_csv(path_labelled_df)

    min_len = get_min_series_len()
    win_len = min_len

    user_id = 'USER_' + str(i)
    if i not in config['general']['excluded_users']:
        user_list = []
        user_label_list = []
        path = config['path']['sync_prefix'] + 'sync_dataset_' + user_id.lower() + '.csv'
        df_sync = pd.read_csv(path)
        for f in config['algorithm']['eeg_features']:
            feature_list = []
            label_list = []
            end_index = - 1
            max_index = int(len(df_sync['time']) / win_len)
            for i in range(0, max_index):
                start_index = end_index + 1
                end_index = start_index + win_len
                arr = np.asarray(df_sync[f].iloc[start_index:end_index]).astype('float32')
                if len(arr) == win_len:
                    feature_list.append(arr)
                    label_index = df_labelled.index[(df_labelled['MEDIA_NAME'] == df_sync['media_name'][start_index]) & (
                                    df_labelled['USER_ID'] == df_sync['user_id'][start_index])]
                    label_list.append(df_labelled['LABEL'][label_index])
            user_list.append(feature_list)
            user_label_list.append(label_list)
        complete_x_list.append(user_list)
        complete_y_list.append(user_label_list)

    return np.array(complete_x_list, dtype=np.ndarray), np.asarray(complete_y_list).astype('int')


def get_users_array():

    complete_x_list = []

    for i in range(1, 53):
        user_id = 'USER_' + str(i)
        if i not in config['general']['excluded_users']:
            user_list = []
            path = config['path']['sync_prefix'] + 'sync_dataset_' + user_id.lower() + '.csv'
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
            path = config['path']['sync_prefix'] + 'sync_dataset_' + user_id.lower() + '.csv'
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


def get_users_oversampled_array():

    complete_x_list = []
    max_len = get_max_series_len()

    for i in range(1, 53):
        user_id = 'USER_' + str(i)
        if i not in config['general']['excluded_users']:
            user_list = []
            path = config['path']['sync_prefix'] + 'sync_dataset_' + user_id.lower() + '.csv'
            df_sync = pd.read_csv(path)
            media_names = df_sync.drop_duplicates('media_name', keep='last')['media_name']
            for name in media_names:
                reduced_df = df_sync[df_sync['media_name'] == name]
                question_list = []
                for f in config['algorithm']['eeg_features']:
                    arr = np.asarray(reduced_df[f]).astype('float32')
                    oversampled_array = numpy.array(0)
                    if config['preprocessing']['resample_library'] == 'sklearn':
                        oversampled_array = sklearn.utils.resample(arr, n_samples=max_len, stratify=arr)
                    elif config['preprocessing']['resample_library'] == 'scipy':
                        oversampled_array = scipy.signal.resample(arr, max_len)
                    question_list.append(oversampled_array)
                user_list.append(question_list)
            complete_x_list.append(user_list)

    return np.array(complete_x_list, dtype=np.ndarray)


def get_users_undersampled_array():

    complete_x_list = []
    min_len = get_min_series_len()

    for i in range(1, 53):
        user_id = 'USER_' + str(i)
        if i not in config['general']['excluded_users']:
            user_list = []
            path = config['path']['sync_prefix'] + 'sync_dataset_' + user_id.lower() + '.csv'
            df_sync = pd.read_csv(path)
            media_names = df_sync.drop_duplicates('media_name', keep='last')['media_name']
            for name in media_names:
                reduced_df = df_sync[df_sync['media_name'] == name]
                question_list = []
                for f in config['algorithm']['eeg_features']:
                    arr = np.asarray(reduced_df[f]).astype('float32')
                    oversampled_array = numpy.array(0)
                    if config['preprocessing']['resample_library'] == 'sklearn':
                        oversampled_array = sklearn.utils.resample(arr, n_samples=min_len, stratify=arr)
                    elif config['preprocessing']['resample_library'] == 'scipy':
                        oversampled_array = scipy.signal.resample(arr, min_len)
                    question_list.append(oversampled_array)
                user_list.append(question_list)
            complete_x_list.append(user_list)

    return np.array(complete_x_list, dtype=np.ndarray)

def get_questions_interpolation_array():

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
                time = np.asarray(reduced_df['time']).astype('float32')
                for f in config['algorithm']['gaze_features']:
                    arr = np.asarray(reduced_df[f]).astype('float32')
                    new_time = np.linspace(time[0], time[len(time)-1], max_len)
                    interp_funct = interp1d(time, arr, kind=config['preprocessing']['interpolation_kind'])
                    interpolated_array = interp_funct(new_time)
                    question_list.append(interpolated_array)
                complete_x_list.append(question_list)

    return np.array(complete_x_list, dtype=np.ndarray)


def get_questions_undersampled_array():

    complete_x_list = []

    min_len = get_min_series_len()

    for i in range(1, 53):
        user_id = 'USER_' + str(i)
        if i not in config['general']['excluded_users']:
            path = config['path']['sync_prefix'] + 'sync_dataset_' + user_id.lower() + '.csv'
            df_sync = pd.read_csv(path)
            media_names = df_sync.drop_duplicates('media_name', keep='last')['media_name']
            for name in media_names:
                question_list = []
                reduced_df = df_sync[df_sync['media_name'] == name]
                for f in config['algorithm']['eeg_features']:
                    arr = np.asarray(reduced_df[f]).astype('float32')
                    oversampled_array = numpy.array(0)
                    if config['preprocessing']['resample_library'] == 'sklearn':
                        oversampled_array = sklearn.utils.resample(arr, n_samples=min_len, stratify=arr)
                    elif config['preprocessing']['resample_library'] == 'scipy':
                        oversampled_array = scipy.signal.resample(arr, min_len)
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
                arr = np.array(df_labelled['LABEL'][c])
                user_list.append(np.expand_dims(arr, axis=(0)))
                c += 1
            complete_y_list.append(user_list)

    return np.asarray(complete_y_list).astype('int')


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

    print(array_x.shape)

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
                for j in range(0, 15):
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