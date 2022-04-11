import statistics.utilities
import pandas as pd


def get_user_answers_df(i):

    user_id = 'USER_' + str(i)
    source = 'datasets/eye-tracker/User ' + str(i) + '_all_gaze.csv'

    df_eye = pd.read_csv(source)
    df_eye = df_eye.drop_duplicates('MEDIA_NAME', keep='last')

    if i == 25 or i == 26 or i == 30:
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
        print(media_name[j])
        if media_name[j] == 'NewMedia7':
            if not (i == 25 or i == 26 or i == 30) and (user_answer[j] == 'a' or user_answer[j] == 'b'):
                n_right_answers += 1
        else:
            if user_answer[j] == correct_answer[j]:
                n_right_answers += 1

    return n_right_answers
