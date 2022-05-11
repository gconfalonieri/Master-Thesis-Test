import pandas as pd
import statistics.utilities
import toml

config = toml.load('config.toml')

pd.options.mode.chained_assignment = None

def get_d_ij(user_id, media_name):

    df_time_for_users = pd.read_csv('datasets/results/times_for_users.csv')
    df_statistics_times_for_questions = pd.read_csv('datasets/results/statistics_times_for_questions.csv')

    i = df_time_for_users[ df_time_for_users['USER_ID'] == user_id ].index[0]
    j = df_statistics_times_for_questions[ df_statistics_times_for_questions['MEDIA_NAME'] == media_name ].index[0]

    t_ij = df_time_for_users[media_name][i]
    av_t_j = df_statistics_times_for_questions['AVERAGE_TIME'][j]

    d_ij = (t_ij - av_t_j) / t_ij

    return d_ij


def get_user_d_ij_list(user_id):

    df_time_for_users = pd.read_csv('datasets/results/times_for_users.csv')
    df_statistics_times_for_questions = pd.read_csv('datasets/results/statistics_times_for_questions.csv')

    i = df_time_for_users[ df_time_for_users['USER_ID'] == user_id ].index[0]

    user_d_ij_list = []

    for media_id in range(1, 25):
        media_name = 'NewMedia' + str(media_id)
        j = df_statistics_times_for_questions[df_statistics_times_for_questions['MEDIA_NAME'] == media_name].index[0]
        t_ij = df_time_for_users[media_name][i]
        av_t_j = df_statistics_times_for_questions['AVERAGE_TIME'][j]
        d_ij = (t_ij - av_t_j) / t_ij
        user_d_ij_list.append(d_ij)

    return user_d_ij_list


def get_av_d_i(user_id):

    n_questions = config['general']['n_questions']
    user_d_ij_list = get_user_d_ij_list(user_id)

    sum = 0

    for d_ij in user_d_ij_list:
        sum += d_ij

    av_d_i = sum / n_questions

    return av_d_i

def get_av_t_j(media_name):

    df_statistics_times_for_questions = pd.read_csv('datasets/results/statistics_times_for_questions.csv')

    j = df_statistics_times_for_questions[ df_statistics_times_for_questions['MEDIA_NAME'] == media_name ].index[0]

    av_t_j = df_statistics_times_for_questions['AVERAGE_TIME'][j]

    return av_t_j

def get_answer_complete_all_info_df(n_users):

    df_answer_complete_all_info = pd.DataFrame(columns=['MEDIA_NAME', 'USER_ID', 'SPECIFIC_CATEGORY', 'MACRO_CATEGORY', 'CORRECT_ANSWER', 'USER_ANSWER', 'T_I_J', 'AVERAGE_T_J', 'D_I_J','AVERAGE_D_I'])

    media_name = []
    user_id_col = []
    specific_category = []
    macro_category = []
    correct_answer = []
    user_answer = []
    col_t_i_j = []
    col_av_t_j = []
    col_d_i_j = []
    col_av_d_i = []

    for i in range(1, n_users):

        if i not in config['general']['excluded_users']:

            user_id = 'USER_' + str(i)
            eye_source = 'datasets/eye-tracker/User ' + str(i) + '_all_gaze.csv'
            user_answers_source = 'datasets/users/answers/answer_user_' + str(i) + '.csv'

            df_user_answer = pd.read_csv(user_answers_source)
            df_categories_complete = pd.read_csv('datasets/questions/categories_complete.csv')

            df_eye_complete = pd.read_csv(eye_source)
            df_eye = df_eye_complete.drop_duplicates('MEDIA_NAME', keep='last')

            av_d_i = get_av_d_i(user_id)

            for x in df_eye['MEDIA_NAME']:
                media_name.append(x)
                user_id_col.append(user_id)
                specific_category.append(statistics.utilities.get_specific_category(x, df_categories_complete))
                macro_category.append(statistics.utilities.get_macro_category(x, df_categories_complete))
                correct_answer.append(statistics.utilities.get_correct_answer(x, df_user_answer))
                user_answer.append(statistics.utilities.get_user_answer(x, df_user_answer))
                col_t_i_j.append(statistics.utilities.get_answer_time(x, df_eye_complete))
                col_av_t_j.append(get_av_t_j(x))
                col_d_i_j.append(get_d_ij(user_id, x))
                col_av_d_i.append(av_d_i)

            print(user_id + " COMPLETE")

    df_answer_complete_all_info['MEDIA_NAME'] = media_name
    df_answer_complete_all_info['USER_ID'] = user_id_col
    df_answer_complete_all_info['SPECIFIC_CATEGORY'] = specific_category
    df_answer_complete_all_info['MACRO_CATEGORY'] = macro_category
    df_answer_complete_all_info['CORRECT_ANSWER'] = correct_answer
    df_answer_complete_all_info['USER_ANSWER'] = user_answer
    df_answer_complete_all_info['T_I_J'] = col_t_i_j
    df_answer_complete_all_info['AVERAGE_T_J'] = col_av_t_j
    df_answer_complete_all_info['D_I_J'] = col_d_i_j
    df_answer_complete_all_info['AVERAGE_D_I'] = col_av_d_i

    return df_answer_complete_all_info
