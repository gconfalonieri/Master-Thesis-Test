import toml
import pandas as pd
from datetime import datetime

config = toml.load('config.toml')

min_norm_value = config['preprocessing']['min_normalization']
max_norm_value = config['preprocessing']['max_normalization']


def normalize_in_range(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr)) * diff) / diff_arr) + t_min
        # temp = (i - min(arr)) / diff_arr
        norm_arr.append(temp)
    return norm_arr


def get_dict_start_seconds(user_id, data_type):
    path = ""

    if data_type == 'eye':
        path = "datasets/timestamps/eye-tracker/timestamp_eye_" + user_id.lower() + '.txt'
    elif data_type == 'eeg':
        path = "datasets/timestamps/eeg/timestamp_eeg_" + user_id.lower() + ".txt"

    file = open(path, "r")
    timestamp = file.readline()
    file.close()
    date = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f")

    seconds = (date.hour * 3600) + (date.minute * 60) + date.second + (date.microsecond / 1000000)

    return seconds


def undersample_gaze_df():
    for i in range(1, config['general']['n_testers']+1):
        if i not in config['general']['excluded_users']:
            df_gaze = pd.read_csv('datasets/eye-tracker/User ' + str(i) + '_all_gaze.csv')
            media_names = df_gaze.drop_duplicates('MEDIA_NAME', keep='last')['MEDIA_NAME']
            undersampled_list = []
            for media in media_names:
                question_df = df_gaze[df_gaze['MEDIA_NAME'] == media]
                undersampled_question_df = question_df[question_df.index % 15 == 0]
                undersampled_list.append(undersampled_question_df)
            undersampled_df = pd.concat(undersampled_list)
            undersampled_df.to_csv('datasets/sync_datasets/undersampled_gaze/User ' + str(i) + '_all_gaze_undersampled.csv')


for i in range(1, 53):
    user_id = 'USER_' + str(i)
    if i not in config['general']['excluded_users']:
        df_gaze = pd.read_csv('datasets/sync_datasets/undersampled_gaze/User ' + str(i) + '_all_gaze_undersampled.csv')
        df_eeg = pd.read_csv('datasets/eeg/eeg_user_' + str(i) + '.csv')
        start_eye = get_dict_start_seconds(user_id, 'eye')
        start_eeg = get_dict_start_seconds(user_id, 'eeg')

        media_list = df_gaze['MEDIA_NAME']
        fpogx_list = df_gaze['FPOGX']
        fpogy_list = df_gaze['FPOGY']
        fpogv_list = df_gaze['FPOGV']
        rpd_list = df_gaze['RPD']
        lpd_list = df_gaze['LPD']

        if config['preprocessing']['sync_normalization']:
            fpogx_list = normalize_in_range(fpogx_list, min_norm_value, max_norm_value)
            fpogy_list = normalize_in_range(fpogy_list, min_norm_value, max_norm_value)
            fpogv_list = normalize_in_range(fpogv_list, min_norm_value, max_norm_value)
            rpd_list = normalize_in_range(rpd_list, min_norm_value, max_norm_value)
            lpd_list = normalize_in_range(lpd_list, min_norm_value, max_norm_value)

        sync_dataframe = pd.DataFrame()

        sync_dataframe['media_name'] = media_list
        sync_dataframe['FPOGX'] = fpogx_list
        sync_dataframe['FPOGY'] = fpogy_list
        sync_dataframe['FPOGV'] = fpogv_list
        sync_dataframe['RPD'] = rpd_list
        sync_dataframe['LPD'] = lpd_list

        sync_dataframe.to_csv(config['path']['sync_prefix'] + 'sync_dataset_user_' + str(i) + '.csv', index=False)

        print(user_id + " DONE")
