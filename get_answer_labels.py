import statistics.utilities
import pandas as pd

df_complete = pd.read_csv('datasets/results/answers_complete_all_info.csv')

answer_dict = statistics.utilities.get_answer_times_dict(df_complete)
mean_dict = statistics.utilities.get_mean_dict(answer_dict)
median_dict = statistics.utilities.get_median_dict(answer_dict)

print(mean_dict)
print(median_dict)