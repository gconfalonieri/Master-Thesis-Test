import statistics.utilities
import pandas as pd

df_complete = pd.read_csv('datasets/results/answers_complete_all_info.csv')

answer_dict = statistics.utilities.get_answer_times_dict(df_complete)
min_dict = statistics.utilities.get_min_right_dict(df_complete)
mean_dict = statistics.utilities.get_mean_dict(answer_dict)
median_dict = statistics.utilities.get_median_dict(answer_dict)

df_answers_labelled_mean = statistics.set_labels.get_df_answers_labelled(df_complete, min_dict, mean_dict)
df_answers_labelled_median = statistics.set_labels.get_df_answers_labelled(df_complete, min_dict, mean_dict)

df_answers_labelled_mean.to_csv('datasets/results/labels_mean_answers_all_info.csv', index=False)
df_answers_labelled_median.to_csv('datasets/results/labels_median_answers_all_info.csv', index=False)
