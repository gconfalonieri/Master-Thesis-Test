import pandas as pd
import toml
import labelling.utilities

config = toml.load('config.toml')

simplest_questions = labelling.utilities.get_n_hardest_questions(3)
df_times_for_users = pd.read_csv('datasets/results/times_for_users.csv')

print(simplest_questions)

sum = 0

for q in simplest_questions:
    simplest_index = df_times_for_users[df_times_for_users['USER_ID'] == 'USER_1'].index[0]
    print(df_times_for_users[q])
    sum += df_times_for_users[q][simplest_index]

av_d_i_f = sum / 3

print(av_d_i_f)