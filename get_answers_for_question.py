import pandas as pd
import os

df_correct_answer = pd.read_csv('questions/solutions_complete.csv')
df_user_answers = pd.read_csv('questions/answers_complete.csv')

df_answers_for_question = pd.DataFrame(columns=['ID', 'MEDIA_NAME', 'CORRECT_ANSWER'])
df_answers_for_question['ID'] = df_user_answers['ID']
df_answers_for_question['MEDIA_NAME'] = df_correct_answer['MEDIA_NAME']
df_answers_for_question['CORRECT_ANSWER'] = df_correct_answer['CORRECT_ANSWER']

list = os.listdir('eye-tracker')
number_files = int(len(list) / 2) + 1

for i in range(1, number_files):

    if not(i==25 or i==26 or i==30):

        user_id = 'USER_' + str(i)
        user_answers = 'users/answers/answer_user_' + str(i) + '.csv'

        df_user = pd.read_csv (user_answers)
        answers = dict.fromkeys(df_correct_answer['MEDIA_NAME'], 0)

        for j in df_user.index:
            answers[df_user['MEDIA_NAME'][j]] = df_user['USER_ANSWER'][j]

        df_answers_for_question[user_id] = answers.values()


df_answers_for_question.to_csv('results/answers_for_question.csv', index=False)