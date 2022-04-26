from numpy import mean, std
import pandas as pd
import toml

config = toml.load('config.toml')


def get_answers_times_df(df_correct_answer, n_users):

    df_times_for_question = pd.DataFrame(columns=['MEDIA_NAME'])
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

    return df_times_for_question


def get_statistics_times_for_questions_df(df_times_for_question, n_users):

    df_average_times_for_questions = pd.DataFrame(columns=['MEDIA_NAME', 'AVERAGE_TIME', 'STANDARD_DEVIATION'])
    df_average_times_for_questions['MEDIA_NAME'] = df_times_for_question['MEDIA_NAME']

    average_times = []
    standard_deviations = []

    for i in df_times_for_question.index:
        list_times = []
        for j in range(1, n_users):
            if j not in config['general']['excluded_users']:
                list_times.append(df_times_for_question['USER_' + str(j)][i])
        average_time = mean(list_times)
        standard_deviation = std(list_times)
        average_times.append(average_time)
        standard_deviations.append(standard_deviation)

    df_average_times_for_questions['AVERAGE_TIME'] = average_times
    df_average_times_for_questions['STANDARD_DEVIATION'] = standard_deviations

    return df_average_times_for_questions
