import pandas as pd
import toml
import labelling.utilities

config = toml.load('config.toml')

simplest_questions = labelling.utilities.get_n_simplest_questions(3)
df_times_for_users = pd.read_csv('datasets/results/times_for_users.csv')
df_questions_statistics = pd.read_csv('datasets/results/questions_statistics.csv')

#print(simplest_questions)

#sum = 0

#for q in simplest_questions:
#    simplest_index = df_times_for_users[df_times_for_users['USER_ID'] == 'USER_1'].index[0]
#    print(df_times_for_users[q])
#    sum += df_times_for_users[q][simplest_index]

# av_d_i_f = sum / 3

# print(av_d_i_f)

sum = 0

print(simplest_questions)

for q in simplest_questions:
    user_index = df_times_for_users[df_times_for_users['USER_ID'] == 'USER_1'].index[0]
    simplest_question_index = df_questions_statistics[df_questions_statistics['MEDIA_NAME'] == q].index[0]
    av_t_f = df_questions_statistics['AVERAGE_TIME'][simplest_question_index]
    t_i_f = df_times_for_users[q][user_index]
    d_i_f = (t_i_f - av_t_f) / t_i_f
    print(d_i_f)
    sum += d_i_f

av_d_i_f = sum / 3