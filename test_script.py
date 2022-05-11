import pandas as pd
import toml
import labelling.utilities

config = toml.load('config.toml')

df_question_statistics = pd.read_csv('datasets/results/questions_statistics.csv')
df_statistics_time_for_questions = pd.read_csv('datasets/results/statistics_times_for_questions.csv')

avg_time = []
std_time = []

for i in df_question_statistics.index:
    avg_time.append(df_statistics_time_for_questions['AVERAGE_TIME'][i])
    std_time.append(df_statistics_time_for_questions['STANDARD_DEVIATION'][i])

df_question_statistics['AVERAGE_TIME'] = df_statistics_time_for_questions['AVERAGE_TIME']
df_question_statistics['STANDARD_DEVIATION'] = df_statistics_time_for_questions['STANDARD_DEVIATION']

print(df_question_statistics)