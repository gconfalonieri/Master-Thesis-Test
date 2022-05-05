import numpy
import matplotlib.pyplot as plt

import statistics.utilities
import toml
import pandas as pd

config = toml.load('config.toml')

n_users = statistics.utilities.get_n_testers() + 1

df_correct_answer = pd.read_csv(config['path']['solutions_complete_dataset'])


def get_total_mean():

    list_times = []

    for i in range(1, n_users):
        if i not in config['general']['excluded_users']:
            user_id = 'USER_' + str(i)
            sum = total_user_time(user_id)
            list_times.append(sum)

    total_mean = numpy.mean(list_times)

    return total_mean


def total_user_time(user_id):

    df_time_for_users = pd.read_csv('datasets/results/times_for_questions.csv')

    sum = 0

    for i in df_time_for_users.index:
        user_question_time = df_time_for_users[user_id][i]
        sum += user_question_time

    return sum


def get_ratio_df(n_users):

    df_statistics_time_for_questions = pd.read_csv('datasets/results/statistics_times_for_questions.csv')
    df_time_for_users = pd.read_csv('datasets/results/times_for_users.csv')

    df_ratio = pd.DataFrame(columns=['USER_ID', 'ANSWER_TIME_16' , 'TOTAL_USER_TIME', 'RATIO_16', 'RATIO_TOTAL'])

    user_list = []
    answer_time_16_list = []
    total_user_time_list = []
    ratio_16_list = []
    ratio_total_list = []

    question_time = df_statistics_time_for_questions['AVERAGE_TIME'][15]
    total_mean_time = get_total_mean()
    counter = 0

    for i in range(1, n_users):
        if i not in config['general']['excluded_users']:
            user_id = 'USER_' + str(i)
            user_question_time = df_time_for_users['NewMedia16'][counter]
            user_total_time = total_user_time(user_id)
            question_16_ratio = user_question_time / question_time
            ratio_total = user_total_time / total_mean_time
            user_list.append(user_id)
            total_user_time_list.append(user_total_time)
            answer_time_16_list.append(user_question_time)
            ratio_16_list.append(question_16_ratio)
            ratio_total_list.append(ratio_total)
            counter += 1

    df_ratio['USER_ID'] = user_list
    df_ratio['ANSWER_TIME_16'] = answer_time_16_list
    df_ratio['TOTAL_USER_TIME'] = total_user_time_list
    df_ratio['RATIO_16'] = ratio_16_list
    df_ratio['RATIO_TOTAL'] = ratio_total_list

    return df_ratio

df_ratio = get_ratio_df(n_users)
df_ratio.to_csv("datasets/results/ratio_df.csv", index=False)

plt.scatter(df_ratio['RATIO_16'], df_ratio['RATIO_TOTAL'])
plt.savefig('test.png')