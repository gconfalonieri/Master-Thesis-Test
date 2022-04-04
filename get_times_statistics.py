import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

pd.options.mode.chained_assignment = None

list = os.listdir('eye-tracker')
number_files = int(len(list) / 2) + 1

df_times_for_question = pd.read_csv('results/times_for_question.csv')
df_questions_statistics = pd.read_csv('results/questions_statistics.csv')

list_times = []

for i in df_times_for_question.index:
    sum = 0
    for j in range(1, number_files):
        if not (j == 25 or j == 26 or j == 30):
            sum = sum + df_times_for_question['USER_'+str(j)][i]
    average_time = sum / (number_files - 4)
    print(number_files - 4)
    list_times.append(average_time)

df_questions_statistics['AVERAGE_TIME'] = list_times

fig = plt.figure()
barWidth = 0.25
br = np.arange(len(list_times))

plt.bar(br, list_times, width = barWidth, edgecolor ='grey', label ='AVERAGE_TIME')
plt.savefig('results/img/average_times.png')

df_questions_statistics.to_csv('results/questions_statistics.csv', index=False)