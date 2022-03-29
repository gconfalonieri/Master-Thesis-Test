import pandas as pd
import os

pd.options.mode.chained_assignment = None

df_correct_answer = pd.read_csv('questions/solutions_complete.csv')
df_user_answers = pd.read_csv('questions/answers_complete.csv')

answers_statistics_dataframe = pd.DataFrame(columns=['ID', 'MEDIA_NAME', 'CORRECT_ANSWER'])
answers_statistics_dataframe['ID'] = df_user_answers['ID']
answers_statistics_dataframe['MEDIA_NAME'] = df_correct_answer['MEDIA_NAME']
answers_statistics_dataframe['CORRECT_ANSWER'] = df_correct_answer['CORRECT_ANSWER']

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

        answers_statistics_dataframe[user_id] = answers.values()


answers_statistics_dataframe.to_csv('results/answers_for_question.csv', index=False)

statistics =  pd.DataFrame(columns=['ID','MEDIA_NAME','TOTAL','RIGHT','WRONG'])
statistics['ID'] = answers_statistics_dataframe['ID']
statistics['MEDIA_NAME'] = answers_statistics_dataframe['MEDIA_NAME']

for i in answers_statistics_dataframe.index:
    statistics['TOTAL'][i] = 0
    statistics['RIGHT'][i] = 0
    statistics['WRONG'][i] = 0

for i in answers_statistics_dataframe.index:
    for j in range(1, number_files):
        if not (j == 25 or j == 26 or j == 30):
            if answers_statistics_dataframe['CORRECT_ANSWER'][i] == answers_statistics_dataframe['USER_'+str(j)][i]:
                statistics['RIGHT'][i] += 1
            else:
                statistics['WRONG'][i] += 1
            statistics['TOTAL'][i] += 1

for i in statistics.index:
    n_right = (statistics['RIGHT'][i] / statistics['TOTAL'][i]) * 100
    n_wrong = (statistics['WRONG'][i] / statistics['TOTAL'][i]) * 100
    print(statistics['MEDIA_NAME'][i] + ' ' + str(n_right) + ' ' + str(n_wrong))