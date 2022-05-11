import pandas as pd
import toml
import labelling.utilities

config = toml.load('config.toml')

df_question_statistics = pd.read_csv('datasets/results/questions_statistics.csv')

print(labelling.utilities.get_n_hardest_questions(3))