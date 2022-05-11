import pandas as pd
import statistics.utilities
import toml

config = toml.load('config.toml')

pd.options.mode.chained_assignment = None


def get_simplest_question():

    simplest_question = ''

    df_question_statistics = pd.read_csv('datasets/results/questions_statistics.csv')

    simplest_thr = df_question_statistics['TOTAL'][0] * 0.25
    min_time = min(df_question_statistics['AVERAGE_TIME'])

    for i in df_question_statistics.index:
        if (df_question_statistics['AVERAGE_TIME'][i] == min_time) and df_question_statistics['RIGHT'][i] > simplest_thr:
            simplest_question = df_question_statistics['MEDIA_NAME'][i]
            break

    return simplest_question


def get_hardest_question():

    hardest_question = ''

    df_question_statistics = pd.read_csv('datasets/results/questions_statistics.csv')

    hardest_thr = df_question_statistics['TOTAL'][0] * 0.25
    max_time = max(df_question_statistics['AVERAGE_TIME'])

    for i in df_question_statistics.index:
        if (df_question_statistics['AVERAGE_TIME'][i] == max_time) and df_question_statistics['WRONG'][i] > hardest_thr:
                hardest_question = df_question_statistics['MEDIA_NAME'][i]
                break

    return hardest_question


def get_n_simplest_questions(n):

    simplest_questions = []

    df_question_statistics = pd.read_csv('datasets/results/questions_statistics.csv')

    simplest_thr = df_question_statistics['TOTAL'][0] * 0.75

    for j in range(0, n):
        min_time = min(df_question_statistics['AVERAGE_TIME'])
        for i in df_question_statistics.index:
            if (df_question_statistics['AVERAGE_TIME'][i] == min_time) and df_question_statistics['RIGHT'][i] > simplest_thr:
                simplest_questions.append(df_question_statistics['MEDIA_NAME'][i])
                df_question_statistics = df_question_statistics.drop(i)

    return simplest_questions


def get_n_hardest_questions(n):

    hardest_questions = []

    df_question_statistics = pd.read_csv('datasets/results/questions_statistics.csv')

    hardest_thr = df_question_statistics['TOTAL'][0] * 0.15

    for j in range(0, n):
        max_time = max(df_question_statistics['AVERAGE_TIME'])
        for i in df_question_statistics.index:
            if (df_question_statistics['AVERAGE_TIME'][i] == max_time) and df_question_statistics['WRONG'][i] > hardest_thr:
                hardest_questions.append(df_question_statistics['MEDIA_NAME'][i])
                df_question_statistics = df_question_statistics.drop(i)

    return hardest_questions


def get_t_i_f(user_id):

    df_question_statistics = pd.read_csv('datasets/results/times_for_users.csv')
    simplest_question = get_simplest_question()
    simplest_index = df_question_statistics[df_question_statistics['USER_ID'] == user_id].index[0]
    t_i_f = df_question_statistics[simplest_question][simplest_index]
    return t_i_f


def get_av_t_f():

    df_questions_statistics = pd.read_csv('datasets/results/questions_statistics.csv')
    simplest_question = get_simplest_question()
    simplest_index = df_questions_statistics[df_questions_statistics['MEDIA_NAME'] == simplest_question].index[0]
    av_t_f = df_questions_statistics['AVERAGE_TIME'][simplest_index]
    return av_t_f


def get_d_i_f(user_id):

    av_t_f = get_av_t_f()
    t_i_f = get_t_i_f(user_id)
    d_i_f = (t_i_f - av_t_f) / t_i_f

    return d_i_f


def get_av_d_i_f(user_id, n):

    simplest_questions = get_n_simplest_questions(n)
    df_times_for_users= pd.read_csv('datasets/results/times_for_users.csv')

    sum = 0

    for q in simplest_questions:
        simplest_index = df_times_for_users[df_times_for_users['USER_ID'] == user_id].index[0]
        sum += df_times_for_users[q][simplest_index]

    av_d_i_f = sum / n

    return av_d_i_f


def get_av_d_i_h(user_id, n):

    hardest_questions = get_n_hardest_questions(n)
    df_times_for_users= pd.read_csv('datasets/results/times_for_users.csv')

    sum = 0

    for q in hardest_questions:
        hardest_index = df_times_for_users[df_times_for_users['USER_ID'] == user_id].index[0]
        sum += df_times_for_users[q][hardest_index]

    av_d_i_h = sum / n

    return av_d_i_h


def get_t_i_h(user_id):

    df_question_statistics = pd.read_csv('datasets/results/times_for_users.csv')
    hardest_question = get_hardest_question()
    hardest_index = df_question_statistics[df_question_statistics['USER_ID'] == user_id].index[0]
    t_i_h = df_question_statistics[hardest_question][hardest_index]
    return t_i_h


def get_av_t_h():

    df_questions_statistics = pd.read_csv('datasets/results/questions_statistics.csv')
    hardest_question = get_simplest_question()
    hardest_index = df_questions_statistics[df_questions_statistics['MEDIA_NAME'] == hardest_question].index[0]
    av_t_h = df_questions_statistics['AVERAGE_TIME'][hardest_index]
    return av_t_h


def get_d_i_h(user_id):

    av_t_h = get_av_t_h()
    t_i_h = get_t_i_h(user_id)
    d_i_h = (t_i_h - av_t_h) / t_i_h

    return d_i_h


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
