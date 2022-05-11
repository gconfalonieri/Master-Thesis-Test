import labelling.set_labels
import statistics.utilities
import pandas as pd

df_complete = pd.read_csv('datasets/results/answers_complete_all_info.csv')


df_labelled = labelling.set_labels.get_df_answers_labelled(df_complete, 'GENERAL_AVERAGE_MEDIA_NAME')
df_label_statistics = labelling.set_labels.get_df_label_statistics(df_labelled, 'MEDIA_NAME')

df_labelled.to_csv('datasets/results/labels/labelled_dataset.csv', index=False)
df_label_statistics.to_csv('datasets/results/labels/label_statistics_dataset', index=False)