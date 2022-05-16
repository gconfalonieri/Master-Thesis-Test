from time import gmtime, strftime

import pandas as pd
import toml
from datetime import datetime
from datetime import datetime
from datetime import timedelta
config = toml.load('config.toml')

df_eye = pd.read_csv('datasets/eye-tracker/User 46_all_gaze.csv')
df_eeg = pd.read_csv('datasets/eeg/eeg_user_46.csv')

eeg_time_col = df_eeg[' time']

eeg_duration = eeg_time_col.iloc[-1]

max_times = dict()

for j in df_eye.index:
    max_times[df_eye['MEDIA_NAME'][j]] = round(df_eye[df_eye.columns[3]][j], 3)

interval_bounds = dict()

eye_duration = 0

for key in max_times:
    eye_duration += max_times[key]


diff = round(eye_duration - eeg_duration, 3)

print(diff)

path_eye = "datasets/timestamps/eye-tracker/timestamp_eye_user_46.txt"
path_eeg = "datasets/timestamps/eeg/timestamp_eeg_user_46.txt"

file_eye = open(path_eye, "r")
timestamp_eye = file_eye.readline()
file_eye.close()
date = datetime.strptime(timestamp_eye, "%Y-%m-%d %H:%M:%S.%f")

print(timestamp_eye)

seconds = (date.hour * 3600) + (date.minute * 60) + date.second + (date.microsecond / 1000000)

print(diff)

seconds_eeg = seconds + diff

print(seconds_eeg)

timestamp_eeg = strftime("%H:%M:%S", gmtime(seconds_eeg))

print(timestamp_eeg)