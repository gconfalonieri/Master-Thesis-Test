import statistics.utilities
import toml
import pandas as pd

config = toml.load('config.toml')

n_users = statistics.utilities.get_n_testers() + 1

df_correct_answer = pd.read_csv(config['path']['solutions_complete_dataset'])

# 1) Compute Answer DataFrame for each User

statistics.questions_statistics.save_all_user_answers_df(n_users)

# 2) Compute DataFrame with all the answers related to a single question

df_answers_for_question = statistics.questions_statistics.get_answer_for_questions_df(df_correct_answer, n_users)
df_answers_for_question.to_csv('datasets/results/answers_for_questions.csv', index=False)
