import pandas as pd
import os

df_correct_answer = pd.read_csv('questions/solutions_complete.csv')
df_user_answers = pd.read_csv('questions/answers_complete.csv')

df_times_for_question = pd.DataFrame(columns=['ID', 'MEDIA_NAME' ])

df_times_for_question['ID'] = df_user_answers['ID']
df_times_for_question['MEDIA_NAME'] = df_correct_answer['MEDIA_NAME']

list = os.listdir('eye-tracker')
number_files = int(len(list) / 2) + 1

for i in range(1, number_files):

    if not(i==25 or i==26 or i==30):

        user_id = 'USER_' + str(i)
        user_eye = 'eye-tracker/User ' + str(i) + '_all_gaze.csv'

        df_user = pd.read_csv(user_eye)
        times = dict.fromkeys(df_correct_answer['MEDIA_NAME'], 0)

        for j in df_user.index:
            times[df_user['MEDIA_NAME'][j]] = df_user.iloc[:, 3][j]

        df_times_for_question[user_id] = times.values()

        print(user_id + " DONE")

df_times_for_question.to_csv('results/times_for_question.csv', index=False)