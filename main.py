import labelling.set_labels
import statistics.utilities
import toml
import pandas as pd

config = toml.load('config.toml')

n_users = statistics.utilities.get_n_testers() + 1

df_correct_answer = pd.read_csv(config['path']['solutions_complete_dataset'])

# 1) Compute Answer DataFrame for each User

# statistics.questions_statistics.save_all_user_answers_df(n_users)

# 2) Compute DataFrame with all the answers related to a single question

# df_answers_for_questions = statistics.questions_statistics.get_answer_for_questions_df(df_correct_answer, n_users)
# df_answers_for_questions.to_csv('datasets/results/answers_for_questions.csv', index=False)

# 3) Compute DataFrame with some statistics related to a single question

# df_questions_statistics = statistics.questions_statistics.get_questions_statistics_df(df_answers_for_questions, n_users)
# df_questions_statistics.to_csv('datasets/results/questions_statistics.csv', index=False)

df_questions_statistics = pd.read_csv('datasets/results/questions_statistics.csv')

statistics.plots.get_questions_statistics_bar_plot(df_questions_statistics)

# 4) Compute DataFrame with times spent for each questions to get each user answer

# df_times_for_question = statistics.time_statistics.get_times_for_question_df(df_correct_answer, n_users)
# df_times_for_question.to_csv("datasets/results/times_for_questions.csv", index=False)

df_times_for_question = pd.read_csv("datasets/results/times_for_questions.csv")

df_times_for_users = statistics.time_statistics.get_times_for_user_df(df_times_for_question, n_users)
df_times_for_users.to_csv("datasets/results/times_for_users.csv", index=False)

# 5) Compute DataFrame with some statistics related to times spent for each questions to get each user answer

df_statistics_times_for_questions = statistics.time_statistics.get_statistics_times_for_questions_df(df_times_for_question, n_users)
df_statistics_times_for_questions.to_csv("datasets/results/statistics_times_for_questions.csv", index=False)
statistics.plots.get_answers_times_statistics_bar_plot(df_statistics_times_for_questions, 'Questions')
statistics.plots.get_answers_times_statistics_bar_plot_normalized(df_statistics_times_for_questions, 'Questions')
statistics.plots.get_error_bar_plot_time_questions(df_statistics_times_for_questions, 2, 'Questions')
statistics.plots.get_error_bar_plot_time_questions_normalized(df_statistics_times_for_questions, 2, 'Questions')

df_statistics_times_for_users = statistics.time_statistics.get_statistics_times_for_users_df(df_times_for_users)
df_statistics_times_for_users.to_csv("datasets/results/statistics_times_for_users.csv", index=False)

statistics.plots.get_answers_times_statistics_bar_plot(df_statistics_times_for_users, 'Users')
statistics.plots.get_answers_times_statistics_bar_plot_normalized(df_statistics_times_for_users, 'Users')
statistics.plots.get_error_bar_plot_time_questions(df_statistics_times_for_users, 5, 'Users')
statistics.plots.get_error_bar_plot_time_questions_normalized(df_statistics_times_for_users, 5, 'Users')

# 6) Compute DataFrame with all the information related to each user answer

# df_answer_complete_all_info = labelling.set_labels.get_answer_complete_all_info_df(n_users)
# df_answer_complete_all_info.to_csv('datasets/results/answers_complete_all_info.csv', index=False)