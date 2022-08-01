import toml
import pandas as pd
from datetime import datetime

import experiments

config = toml.load('config.toml')
experiments.utilities.fix_seeds()

min_norm_value = config['preprocessing']['min_normalization']
max_norm_value = config['preprocessing']['max_normalization']


def normalize_gaze_in_range(f, df_gaze):

    diff = max_norm_value - min_norm_value

    min_fpogx = min(df_gaze['FPOGX'])
    max_fpogx = max(df_gaze['FPOGX'])
    min_fpogy = min(df_gaze['FPOGY'])
    max_fpogy = max(df_gaze['FPOGY'])
    # min_fpogv = min(df_gaze['FPOGV'])
    # max_fpogv = max(df_gaze['FPOGV'])
    min_rpd = min(df_gaze['RPD'])
    max_rpd = max(df_gaze['RPD'])
    min_lpd = min(df_gaze['LPD'])
    max_lpd = max(df_gaze['LPD'])

    diff_fpogx = max_fpogx - min_fpogx
    diff_fpogy = max_fpogy - min_fpogy
    # diff_fpogv = max_fpogv - min_fpogv
    diff_rpd = max_rpd - min_rpd
    diff_lpd = max_lpd - min_lpd

    for i in df_gaze.index:

        norm_fpogx = (((df_gaze['FPOGX'][i] - min_fpogx) * diff) / diff_fpogx) + min_norm_value
        norm_fpogy = (((df_gaze['FPOGY'][i] - min_fpogy) * diff) / diff_fpogy) + min_norm_value
        # norm_fpogv = int((((df_gaze['FPOGV'][i] - min_fpogv) * diff) / diff_fpogv) + min_norm_value)
        norm_rpd = (((df_gaze['RPD'][i] - min_rpd) * diff) / diff_rpd) + min_norm_value
        norm_lpd = (((df_gaze['LPD'][i] - min_lpd) * diff) / diff_lpd) + min_norm_value

        line = df_gaze['MEDIA_NAME'][i] + ',' + str(df_gaze['CNT'][i]) + ',' + \
               str(norm_fpogx) + ',' + str(norm_fpogy) + ',' + str(df_gaze['FPOGV'][i]) + \
               ',' + str(norm_rpd) + ',' + str(norm_lpd) + '\n'

        f.write(line)

def normalize_in_range_int(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr)) * diff) / diff_arr) + t_min
        # temp = (i - min(arr)) / diff_arr
        norm_arr.append(int(temp))
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
    if i not in config['general']['not_valid_users']:
        df_gaze = pd.read_csv('datasets/eye-tracker/User ' + str(i) + '_all_gaze.csv')
        df_eeg = pd.read_csv('datasets/eeg/eeg_user_' + str(i) + '.csv')
        start_eye = get_dict_start_seconds(user_id, 'eye')
        start_eeg = get_dict_start_seconds(user_id, 'eeg')

        f = open(config['path']['sync_prefix'] + 'sync_dataset_user_' + str(i) + '.csv', 'w')
        f.write('media_name,CNT,FPOGX,FPOGY,FPOGV,RPD,LPD\n')
        normalize_gaze_in_range(f, df_gaze)
        f.close()

        # alpha1_list = df_eeg[' Alpha1']
        # alpha2_list = df_eeg[' Alpha2']
        # beta1_list = df_eeg[' Beta1']
        # beta2_list = df_eeg[' Beta2']
        # gamma1_list = df_eeg[' Gamma1']
        # gamma2_list = df_eeg[' Gamma2']
        # theta_list = df_eeg[' Theta']
        # delta_list = df_eeg[' Delta']

        print('# # # - ' + user_id + " DONE - # # #")
