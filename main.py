import experiments
import labelling.set_labels
import statistics.utilities
import toml
import pandas as pd

config = toml.load('config.toml')
experiments.utilities.fix_seeds()

n_users = statistics.utilities.get_n_testers() + 1

df_correct_answer = pd.read_csv(config['path']['solutions_complete_dataset'])

# 6) Compute DataFrame with all the information related to each user answer

df_answer_complete_all_info = labelling.utilities.get_answer_complete_all_info_df(n_users)
df_answer_complete_all_info.to_csv('datasets/results/answers_complete_all_info.csv', index=False)