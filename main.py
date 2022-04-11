import statistics.utilities

n_users = statistics.utilities.get_n_testers() + 1

for i in range(1, n_users):
    destination = 'datasets/users/answers/answer_user_' + str(i) + '.csv'
    answer_dataframe = statistics.questions_statistics.get_user_answers_df(i)
    n_right_answers = statistics.questions_statistics.get_n_right_answers(answer_dataframe, i)
    if i == 25 or i == 26 or i == 30:
        print('USER_' + str(i) + ' Correct answers: ' + str(n_right_answers) + ' / 16')
    else:
        print('USER_' + str(i) + ' Correct answers: ' + str(n_right_answers) + ' / 24')
    answer_dataframe.to_csv(destination, index=False)

