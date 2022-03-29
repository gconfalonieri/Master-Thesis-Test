import pandas as pd
import os

pd.options.mode.chained_assignment = None

list = os.listdir('eye-tracker')
number_files = int(len(list) / 2) + 1

df_answers_for_question = pd.read_csv('results/answers_for_question.csv')

df_questions_statistics = pd.DataFrame(columns=['ID','MEDIA_NAME','TOTAL','RIGHT','WRONG'])
df_questions_statistics['ID'] = df_answers_for_question['ID']
df_questions_statistics['MEDIA_NAME'] = df_answers_for_question['MEDIA_NAME']

for i in df_answers_for_question.index:
    df_questions_statistics['TOTAL'][i] = 0
    df_questions_statistics['RIGHT'][i] = 0
    df_questions_statistics['WRONG'][i] = 0

for i in df_answers_for_question.index:
    for j in range(1, number_files):
        if not (j == 25 or j == 26 or j == 30):
            if df_answers_for_question['CORRECT_ANSWER'][i] == df_answers_for_question['USER_'+str(j)][i]:
                df_questions_statistics['RIGHT'][i] += 1
            else:
                df_questions_statistics['WRONG'][i] += 1
            df_questions_statistics['TOTAL'][i] += 1

for i in df_questions_statistics.index:
    n_right = (df_questions_statistics['RIGHT'][i] / df_questions_statistics['TOTAL'][i]) * 100
    n_wrong = (df_questions_statistics['WRONG'][i] / df_questions_statistics['TOTAL'][i]) * 100
    print(df_questions_statistics['MEDIA_NAME'][i] + ' ' + str(n_right) + ' ' + str(n_wrong))

df_questions_statistics.to_csv('results/questions_statistics.csv', index=False)