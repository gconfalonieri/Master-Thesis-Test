import statistics.utilities
import pandas as pd
import toml

config = toml.load('config.toml')

def get_user_answers_df(i):

    user_id = 'USER_' + str(i)
    source = 'datasets/eye-tracker/User ' + str(i) + '_all_gaze.csv'

    df_eye = pd.read_csv(source)
    df_eye = df_eye.drop_duplicates('MEDIA_NAME', keep='last')

    if i in config['general']['reduced_test_users']:
        df_correct_answer = pd.read_csv('datasets/questions/solutions_reduced.csv')
        df_user_answer = pd.read_csv('datasets/questions/answers_reduced.csv')
    else:
        df_correct_answer = pd.read_csv('datasets/questions/solutions_complete.csv')
        df_user_answer = pd.read_csv('datasets/questions/answers_complete.csv')

    answer_dataframe = pd.DataFrame(columns=['ID', 'MEDIA_NAME', 'USER_ANSWER', 'CORRECT_ANSWER'])

    answer_dataframe['ID'] = statistics.utilities.get_index_column(len(df_user_answer[user_id]))
    answer_dataframe['MEDIA_NAME'] = statistics.utilities.get_media_name_column(df_eye)
    answer_dataframe['USER_ANSWER'] = df_user_answer[user_id]
    answer_dataframe['CORRECT_ANSWER'] = statistics.utilities.correct_answer_column(df_eye, df_correct_answer)

    return answer_dataframe


def get_n_right_answers(df_user_answers, i):

    n_right_answers = 0
    user_answer = df_user_answers['USER_ANSWER']
    media_name = df_user_answers['MEDIA_NAME']
    correct_answer = df_user_answers['CORRECT_ANSWER']

    for j in range(0, len(user_answer)):
        if media_name[j] == 'NewMedia7':
            if i not in config['general']['reduced_test_users'] and (user_answer[j] == 'a' or user_answer[j] == 'b'):
                n_right_answers += 1
        else:
            if user_answer[j] == correct_answer[j]:
                n_right_answers += 1

    return n_right_answers


def save_all_user_answers_df(n_users):
    for i in range(1, n_users):
        destination = 'datasets/users/answers/answer_user_' + str(i) + '.csv'
        answer_dataframe = statistics.questions_statistics.get_user_answers_df(i)
        n_right_answers = statistics.questions_statistics.get_n_right_answers(answer_dataframe, i)
        if i in config['general']['reduced_test_users']:
            print('USER_' + str(i) + ' Correct answers: ' + str(n_right_answers) + ' / 16')
        else:
            print('USER_' + str(i) + ' Correct answers: ' + str(n_right_answers) + ' / 24')
        answer_dataframe.to_csv(destination, index=False)


def get_answer_for_questions_df(df_correct_answers, n_users):

    df_answers_for_question = pd.DataFrame(columns=['MEDIA_NAME', 'CORRECT_ANSWER'])
    df_answers_for_question['MEDIA_NAME'] = df_correct_answers['MEDIA_NAME']
    df_answers_for_question['CORRECT_ANSWER'] = df_correct_answers['CORRECT_ANSWER']

    for i in range(1, n_users):

        if i not in config['general']['excluded_users']:

            user_id = 'USER_' + str(i)
            user_answers = 'datasets/users/answers/answer_user_' + str(i) + '.csv'

            df_user = pd.read_csv(user_answers)
            answers = dict.fromkeys(df_correct_answers['MEDIA_NAME'], 0)

            for j in df_user.index:
                answers[df_user['MEDIA_NAME'][j]] = df_user['USER_ANSWER'][j]

            df_answers_for_question[user_id] = answers.values()

    return df_answers_for_question


def get_questions_statistics_df(df_answers_for_questions, n_users):

    df_questions_statistics = pd.DataFrame(columns=['MEDIA_NAME', 'TOTAL', 'RIGHT', 'WRONG'])
    df_questions_statistics['MEDIA_NAME'] = df_answers_for_questions['MEDIA_NAME']

    for i in df_answers_for_questions.index:
        df_questions_statistics['TOTAL'][i] = 0
        df_questions_statistics['RIGHT'][i] = 0
        df_questions_statistics['WRONG'][i] = 0

    for i in df_answers_for_questions.index:
        for j in range(1, n_users):
            if j not in config['general']['excluded_users']:
                if df_answers_for_questions['CORRECT_ANSWER'][i] == df_answers_for_questions['USER_' + str(j)][i]:
                    df_questions_statistics['RIGHT'][i] += 1
                else:
                    df_questions_statistics['WRONG'][i] += 1
                df_questions_statistics['TOTAL'][i] += 1

    return df_questions_statistics
