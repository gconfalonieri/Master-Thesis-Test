from datetime import datetime

import pandas as pd
import toml
import matplotlib.pyplot as plt
import labelling.utilities

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


config = toml.load('config.toml')


for i in range(1, 53):
    user_id = 'USER_' + str(i)
    if i not in config['general']['excluded_users']:
        df_eye = pd.read_csv('datasets/eye-tracker/User ' + str(i) + '_all_gaze.csv')
        df_eeg = pd.read_csv('datasets/eeg/eeg_user_' + str(i) + '.csv')
        media_names = df_eye.drop_duplicates('MEDIA_NAME', keep='last')['MEDIA_NAME']
        eye_time = df_eye[df_eye.columns[3]]
        eeg_time_col = df_eeg[' time']
        total_power_channel = df_eeg[' totPwr']
        attention_channel = df_eeg[' Attention']
        meditation_channel = df_eeg[' Meditation']
        FPOGX_serie = df_eye['FPOGX']
        eeg_time = []
        start_eye = get_dict_start_seconds(user_id, 'eye')
        start_eeg = get_dict_start_seconds(user_id, 'eeg')

        for x in eeg_time_col:
            time = x + start_eeg
            eeg_time.append(time)

        # compute duration of each media

        max_times = dict()

        for j in df_eye.index:
            max_times[df_eye['MEDIA_NAME'][j]] = int(df_eye[df_eye.columns[3]][j])

        # compute upper bound of each media

        interval_bounds = dict()

        sum = start_eye

        for key in max_times:
            interval_bounds[key] = sum
            sum += max_times[key]

        # compute fpogx starting from zero to the last sampling time

        all_fpogx = dict()
        all_media_name = dict()
        fpogx_list = []
        media_list = []
        user_id_list = []

        for j in df_eye.index:
            index = int(df_eye[df_eye.columns[3]][j]) + interval_bounds[df_eye['MEDIA_NAME'][j]]
            all_fpogx[index] = df_eye['FPOGX'][j]
            all_media_name[index] = df_eye['MEDIA_NAME'][j]

        for key1 in eeg_time:
            for key2 in all_fpogx:
                if int(key1) == int(key2):
                    fpogx_list.append(all_fpogx[key2])
                    media_list.append(all_media_name[key2])
                    user_id_list.append(user_id)

        # adapt column to the shortest duration

        reduced_eeg_time = []
        reduced_delta = []
        reduced_alpha_1 = []
        reduced_alpha_2 = []
        reduced_beta1 = []
        reduced_beta2 = []
        reduced_gamma1 = []
        reduced_gamma2 = []
        reduced_theta = []
        reduced_totPwr = []



        for j in range(0, len(fpogx_list)):
            reduced_eeg_time.append(round((eeg_time[j] - start_eeg), 1))
            reduced_alpha_1.append(df_eeg[' Alpha1'][j])
            reduced_alpha_2.append(df_eeg[' Alpha2'][j])
            reduced_delta.append(df_eeg[' Delta'][j])
            reduced_beta1.append(df_eeg[' Beta1'][j])
            reduced_beta2.append(df_eeg[' Beta2'][j])
            reduced_gamma1.append(df_eeg[' Gamma1'][j])
            reduced_gamma2.append(df_eeg[' Gamma2'][j])
            reduced_theta.append(df_eeg[' Theta'][j])
            reduced_totPwr.append(total_power_channel[j])

        # save dataframe

        sync_dataframe = pd.DataFrame(columns=['time', 'user_id', 'media_name', 'delta', 'alpha1', 'alpha2', 'beta1', 'beta2',
                                               'gamma1', 'gamma2', 'theta', 'totalPower', 'FPOGX'])

        sync_dataframe['time'] = reduced_eeg_time

        # min_delta = min(reduced_delta)
        # max_delta = max(reduced_delta)

        # for value in reduced_delta:
        #    delta_list.append(round(((value - min_delta) / (max_delta - min_delta)),5))

        sync_dataframe['user_id'] = user_id_list
        sync_dataframe['media_name'] = media_list
        sync_dataframe['delta'] = reduced_delta
        sync_dataframe['alpha1'] = reduced_alpha_1
        sync_dataframe['alpha2'] = reduced_alpha_2
        sync_dataframe['beta1'] = reduced_beta1
        sync_dataframe['beta2'] = reduced_beta2
        sync_dataframe['gamma1'] = reduced_gamma1
        sync_dataframe['gamma2'] = reduced_gamma2
        sync_dataframe['theta'] = reduced_theta
        sync_dataframe['totalPower'] = reduced_totPwr
        sync_dataframe['FPOGX'] = fpogx_list

        sync_dataframe.to_csv('datasets/sync_datasets/sync_dataset_user_' + str(i) + '.csv', index=False)

        print(user_id + " DONE")

        # save plot

        # plt.plot(sync_dataframe['time'], sync_dataframe['delta'], color='r', label='normalized delta channel')
        # plt.plot(sync_dataframe['time'], sync_dataframe['FPOGX'], color='g', label='fixation point-of-gaze x')
        # plt.legend(loc="upper left")
        # plt.savefig('test_plot.png')