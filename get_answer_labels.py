import labelling.set_labels
import statistics.utilities
import pandas as pd

df_complete = pd.read_csv('datasets/results/answers_complete_all_info.csv')

n = 65

# LABEL BASED ON EACH ANSWER

answer_dict = statistics.utilities.get_times_dict(df_complete, 'MEDIA_NAME')
min_answer_dict = statistics.utilities.get_min_right_dict(df_complete, 'MEDIA_NAME')
mean_answer_dict = statistics.utilities.get_mean_dict(answer_dict)
median_answer_dict = statistics.utilities.get_median_dict(answer_dict)
percentile_answer_dict = statistics.utilities.get_percentile_dict(answer_dict, n)

df_answers_labelled_mean = labelling.set_labels.get_df_answers_labelled(df_complete, min_answer_dict, mean_answer_dict, 'MEDIA_NAME')
df_answers_labelled_median = labelling.set_labels.get_df_answers_labelled(df_complete, min_answer_dict, median_answer_dict, 'MEDIA_NAME')
df_answers_labelled_percentile = labelling.set_labels.get_df_answers_labelled(df_complete, min_answer_dict, percentile_answer_dict, 'MEDIA_NAME')

df_labelled_mean_answer_statistics = labelling.set_labels.get_df_label_statistics(df_answers_labelled_mean, 'MEDIA_NAME')
df_labelled_median_answer_statistics = labelling.set_labels.get_df_label_statistics(df_answers_labelled_median, 'MEDIA_NAME')
df_labelled_percentile_answer_statistics = labelling.set_labels.get_df_label_statistics(df_answers_labelled_percentile, 'MEDIA_NAME')

df_answers_labelled_mean.to_csv('datasets/results/labels/mean/labels_mean_answers_all_info.csv', index=False)
df_answers_labelled_median.to_csv('datasets/results/labels/median/labels_median_answers_all_info.csv', index=False)
df_answers_labelled_percentile.to_csv('datasets/results/labels/percentile/labels_'+ str(n) +'tile_answers_all_info.csv', index=False)

df_labelled_mean_answer_statistics.to_csv('datasets/results/labels/statistic/labels_mean_statistics_answers_all_info.csv', index=False)
df_labelled_median_answer_statistics.to_csv('datasets/results/labels/statistic/labels_median_statistics_answers_all_info.csv', index=False)
df_labelled_percentile_answer_statistics.to_csv('datasets/results/labels/statistic/labels_'+ str(n) +'tile_statistics_answers_all_info.csv', index=False)


statistics.plots.get_labels_pie_plot(df_labelled_mean_answer_statistics, 'MEDIA_NAME', 'plots/mean_labels/answers/')
statistics.plots.get_total_labels_pie_plot(df_labelled_mean_answer_statistics, 'plots/mean_labels/answers/')
statistics.plots.get_labels_pie_plot(df_labelled_median_answer_statistics, 'MEDIA_NAME', 'plots/median_labels/answers/')
statistics.plots.get_total_labels_pie_plot(df_labelled_median_answer_statistics, 'plots/median_labels/answers/')
statistics.plots.get_labels_pie_plot(df_labelled_percentile_answer_statistics, 'MEDIA_NAME', 'plots/percentile_labels/answers/')
statistics.plots.get_total_labels_pie_plot(df_labelled_percentile_answer_statistics, 'plots/percentile_labels/answers/')

# LABEL BASED ON SPECIFIC CATEGORIES

specific_categories_dict = statistics.utilities.get_times_dict(df_complete, 'SPECIFIC_CATEGORY')
min_specific_categories_dict = statistics.utilities.get_min_right_dict(df_complete, 'SPECIFIC_CATEGORY')
mean_specific_categories_dict = statistics.utilities.get_mean_dict(specific_categories_dict)
median_specific_categories_dict = statistics.utilities.get_median_dict(specific_categories_dict)
percentile_specific_categories_dict = statistics.utilities.get_percentile_dict(specific_categories_dict, n)

df_specific_categories_labelled_mean = labelling.set_labels.get_df_answers_labelled(df_complete, min_specific_categories_dict, mean_specific_categories_dict, 'SPECIFIC_CATEGORY')
df_specific_categories_labelled_median = labelling.set_labels.get_df_answers_labelled(df_complete, min_specific_categories_dict, median_specific_categories_dict, 'SPECIFIC_CATEGORY')
df_specific_categories_labelled_percentile = labelling.set_labels.get_df_answers_labelled(df_complete, min_specific_categories_dict, percentile_specific_categories_dict, 'SPECIFIC_CATEGORY')

df_labelled_specific_categories_mean_statistics = labelling.set_labels.get_df_label_statistics(df_specific_categories_labelled_mean, 'SPECIFIC_CATEGORY')
df_labelled_specific_categories_median_statistics = labelling.set_labels.get_df_label_statistics(df_specific_categories_labelled_median, 'SPECIFIC_CATEGORY')
df_labelled_specific_categories_percentile_statistics = labelling.set_labels.get_df_label_statistics(df_specific_categories_labelled_percentile, 'SPECIFIC_CATEGORY')

df_specific_categories_labelled_mean.to_csv('datasets/results/labels/mean/labels_mean_specific_categories_all_info.csv', index=False)
df_specific_categories_labelled_median.to_csv('datasets/results/labels/median/labels_median_specific_categories_all_info.csv', index=False)
df_specific_categories_labelled_percentile.to_csv('datasets/results/labels/percentile/labels_'+ str(n) +'tile_specific_categories_all_info.csv', index=False)

df_labelled_specific_categories_mean_statistics.to_csv('datasets/results/labels/statistic/labels_mean_statistics_specific_categories_all_info.csv', index=False)
df_labelled_specific_categories_median_statistics.to_csv('datasets/results/labels/statistic/labels_median_statistics_specific_categories_all_info.csv', index=False)
df_labelled_specific_categories_percentile_statistics.to_csv('datasets/results/labels/statistic/labels_'+ str(n) +'tile_statistics_specific_categories_all_info.csv', index=False)

statistics.plots.get_labels_pie_plot(df_labelled_specific_categories_mean_statistics, 'SPECIFIC_CATEGORY', 'plots/mean_labels/specific_categories/')
statistics.plots.get_total_labels_pie_plot(df_labelled_specific_categories_mean_statistics, 'plots/mean_labels/specific_categories/')
statistics.plots.get_labels_pie_plot(df_labelled_specific_categories_median_statistics, 'SPECIFIC_CATEGORY', 'plots/median_labels/specific_categories/')
statistics.plots.get_total_labels_pie_plot(df_labelled_specific_categories_median_statistics, 'plots/median_labels/specific_categories/')
statistics.plots.get_labels_pie_plot(df_labelled_specific_categories_percentile_statistics, 'SPECIFIC_CATEGORY', 'plots/percentile_labels/specific_categories/')
statistics.plots.get_total_labels_pie_plot(df_labelled_specific_categories_percentile_statistics, 'plots/percentile_labels/specific_categories/')

# LABEL BASED ON MACRO CATEGORIES

macro_categories_dict = statistics.utilities.get_times_dict(df_complete, 'MACRO_CATEGORY')
min_macro_categories_dict = statistics.utilities.get_min_right_dict(df_complete, 'MACRO_CATEGORY')
mean_macro_categories_dict = statistics.utilities.get_mean_dict(macro_categories_dict)
median_macro_categories_dict = statistics.utilities.get_median_dict(macro_categories_dict)
percentile_macro_categories_dict = statistics.utilities.get_percentile_dict(macro_categories_dict, n)

df_macro_categories_labelled_mean = labelling.set_labels.get_df_answers_labelled(df_complete, min_macro_categories_dict, mean_macro_categories_dict, 'MACRO_CATEGORY')
df_macro_categories_labelled_median = labelling.set_labels.get_df_answers_labelled(df_complete, min_macro_categories_dict, median_macro_categories_dict, 'MACRO_CATEGORY')
df_macro_categories_labelled_percentile = labelling.set_labels.get_df_answers_labelled(df_complete, min_macro_categories_dict, percentile_macro_categories_dict, 'MACRO_CATEGORY')

df_labelled_macro_categories_mean_statistics = labelling.set_labels.get_df_label_statistics(df_macro_categories_labelled_mean, 'MACRO_CATEGORY')
df_labelled_macro_categories_median_statistics = labelling.set_labels.get_df_label_statistics(df_macro_categories_labelled_median, 'MACRO_CATEGORY')
df_labelled_macro_categories_percentile_statistics = labelling.set_labels.get_df_label_statistics(df_macro_categories_labelled_percentile, 'MACRO_CATEGORY')

df_macro_categories_labelled_mean.to_csv('datasets/results/labels/mean/labels_mean_macro_categories_all_info.csv', index=False)
df_macro_categories_labelled_median.to_csv('datasets/results/labels/median/labels_median_macro_categories_all_info.csv', index=False)
df_macro_categories_labelled_percentile.to_csv('datasets/results/labels/percentile/labels_'+ str(n) +'tile_macro_categories_all_info.csv', index=False)

df_labelled_macro_categories_mean_statistics.to_csv('datasets/results/labels/statistic/labels_mean_statistics_macro_categories_all_info.csv', index=False)
df_labelled_macro_categories_median_statistics.to_csv('datasets/results/labels/statistic/labels_median_statistics_macro_categories_all_info.csv', index=False)
df_labelled_macro_categories_percentile_statistics.to_csv('datasets/results/labels/statistic/labels_'+ str(n) +'tile_statistics_macro_categories_all_info.csv', index=False)

statistics.plots.get_labels_pie_plot(df_labelled_macro_categories_mean_statistics, 'MACRO_CATEGORY', 'plots/mean_labels/macro_categories/')
statistics.plots.get_total_labels_pie_plot(df_labelled_macro_categories_mean_statistics, 'plots/mean_labels/macro_categories/')
statistics.plots.get_labels_pie_plot(df_labelled_macro_categories_median_statistics, 'MACRO_CATEGORY', 'plots/median_labels/macro_categories/')
statistics.plots.get_total_labels_pie_plot(df_labelled_macro_categories_median_statistics, 'plots/median_labels/macro_categories/')
statistics.plots.get_labels_pie_plot(df_labelled_macro_categories_percentile_statistics, 'MACRO_CATEGORY', 'plots/percentile_labels/macro_categories/')
statistics.plots.get_total_labels_pie_plot(df_labelled_macro_categories_percentile_statistics, 'plots/percentile_labels/macro_categories/')