import pandas as pd
import toml

config = toml.load('config.toml')


def get_answers_times_df(df_correct_answer, n_users):

    df_times_for_question = pd.DataFrame(columns=['ID', 'MEDIA_NAME'])
    df_times_for_question['MEDIA_NAME'] = df_correct_answer['MEDIA_NAME']

    for i in range(1, n_users):

        if i not in config['general']['excluded_users']:

            user_id = 'USER_' + str(i)
            user_eye = 'datasets/eye-tracker/User ' + str(i) + '_all_gaze.csv'

            df_user = pd.read_csv(user_eye)
            times = dict.fromkeys(df_correct_answer['MEDIA_NAME'], 0)

            for j in range(1, 25):
                df_media = df_user[df_user['MEDIA_NAME'] == ('NewMedia' + str(j))]
                times[('NewMedia' + str(j))] = df_media.iloc[-1, 3]

            df_times_for_question[user_id] = times.values()

            print(user_id + " DONE")

    return  df_times_for_question