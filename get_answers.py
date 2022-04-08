import pandas as pd
import os

list = os.listdir('datasets/eye-tracker')
number_files = int(len(list) / 2) + 1

for i in range(1, number_files):
    user_id = 'USER_' + str(i)
    source = 'eye-tracker/User ' + str(i) + '_all_gaze.csv'
    destination = 'users/answers/answer_user_' + str(i) + '.csv'

    df_eye = pd.read_csv (source)
    df_eye = df_eye.drop_duplicates('MEDIA_NAME', keep='last')

    if i == 25 or i == 26 or i == 30:
        df_correct_answer = pd.read_csv('datasets/questions/solutions_reduced.csv')
        df_user_answer = pd.read_csv('datasets/questions/answers_reduced.csv')
    else:
        df_correct_answer = pd.read_csv('datasets/questions/solutions_complete.csv')
        df_user_answer = pd.read_csv('datasets/questions/answers_complete.csv')

    index = []
    media_name = []
    user_answer = df_user_answer[user_id]
    correct_answer = []

    for j in range(1, len(user_answer) + 1):
        index.append(i)

    for x in df_eye['MEDIA_NAME']:
        media_name.append(x)
        for j in df_correct_answer.index:
            if x == df_correct_answer['MEDIA_NAME'][j]:
                correct_answer.append(df_correct_answer['CORRECT_ANSWER'][j])
                break

    answer_dataframe = pd.DataFrame(columns=['ID', 'MEDIA_NAME', 'USER_ANSWER', 'CORRECT_ANSWER'])

    answer_dataframe['ID'] = index
    answer_dataframe['MEDIA_NAME'] = media_name
    answer_dataframe['USER_ANSWER'] = user_answer
    answer_dataframe['CORRECT_ANSWER'] = correct_answer

    answer_dataframe.to_csv(destination, index=False)

    right_answers = 0

    for j in range(0, len(user_answer)):
        if media_name[j] == 'NewMedia7':
            if not(i == 25 or i == 26 or i == 30) and (user_answer[j]=='a' or user_answer[j]=='b'):
                right_answers += 1
        else:
            if user_answer[j] == correct_answer[j]:
                right_answers += 1

    if i == 25 or i == 26 or i == 30:
        print('USER_' + str(i) + ' Correct answers: ' + str(right_answers) + ' / 16')
    else:
        print('USER_' + str(i) + ' Correct answers: ' + str(right_answers) + ' / 24')
