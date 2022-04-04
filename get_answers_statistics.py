import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

pd.options.mode.chained_assignment = None

list = os.listdir('eye-tracker')
number_files = int(len(list) / 2) + 1

df_answers_for_question = pd.read_csv('results/answers_for_question.csv')

df_questions_statistics = pd.DataFrame(columns=['ID','MEDIA_NAME','TOTAL','RIGHT','WRONG'])
df_questions_statistics['ID'] = df_answers_for_question['ID']
df_questions_statistics['MEDIA_NAME'] = df_answers_for_question['MEDIA_NAME']

list_rights = []
list_wrongs = []

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
    list_rights.append(n_right)
    list_wrongs.append(n_wrong)
    print(df_questions_statistics['MEDIA_NAME'][i] + ' ' + str(n_right) + ' ' + str(n_wrong))

print(list_rights)
print(list_wrongs)

barWidth = 0.25
br1 = np.arange(len(list_rights))
br2 = [x + barWidth for x in br1]

plt.bar(br1, list_rights, color ='g', width = barWidth, edgecolor ='grey', label ='RIGHT')
plt.bar(br2, list_wrongs, color ='r', width = barWidth, edgecolor ='grey', label ='WRONG')

plt.legend()
plt.savefig('results/img/questions_statistics.png')

df_questions_statistics.to_csv('results/questions_statistics.csv', index=False)