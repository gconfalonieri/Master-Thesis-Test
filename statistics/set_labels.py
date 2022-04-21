import pandas as pd
import statistics.utilities

pd.options.mode.chained_assignment = None


def get_df_answers_labelled(df_complete, min_times_dict, threshold_dict, col_name):

    df_complete_labelled = df_complete
    labels = []

    for i in df_complete.index:
        is_right_answer = statistics.utilities.get_answer_right_or_wrong(df_complete_labelled['MEDIA_NAME'][i], df_complete_labelled['USER_ANSWER'][i], df_complete_labelled['CORRECT_ANSWER'][i])
        answer_time = df_complete_labelled['ANSWER_TIME'][i]
        current_min_time = min_times_dict[df_complete_labelled[col_name][i]]
        current_threshold = threshold_dict[df_complete_labelled[col_name][i]]
        if not is_right_answer and answer_time < current_min_time:
            labels.append('NOT_COGNITIVE_EFFORT')
        elif not is_right_answer and answer_time >= current_min_time:
            labels.append('COGNITIVE_EFFORT')
        elif is_right_answer and answer_time > current_threshold:
            labels.append('COGNITIVE_EFFORT')
        elif is_right_answer and current_min_time <= answer_time <= current_threshold:
            labels.append('NOT_COGNITIVE_EFFORT')

    df_complete_labelled['LABEL'] = labels

    return df_complete_labelled


def get_df_label_statistics(df_labelled, col_name):

    df_label_statistics = pd.DataFrame(columns=[col_name, 'N_COGNITIVE_EFFORT', 'N_NOT_COGNITIVE_EFFORT'])
    if col_name == 'MEDIA_NAME':
        df_label_statistics[col_name] = (pd.read_csv('datasets/questions/solutions_complete.csv'))[col_name]
    else:
        df_label_statistics[col_name] = (pd.read_csv('datasets/questions/categories_complete.csv').drop_duplicates(col_name, keep='last'))[col_name]

    effort_dict = dict()
    not_effort_dict = dict()

    for x in df_label_statistics[col_name]:
        effort_dict[x] = 0
        not_effort_dict[x] = 0

    for i in df_labelled.index:
        if df_labelled['LABEL'][i] == "COGNITIVE_EFFORT":
            effort_dict[df_labelled[col_name][i]] += 1
        elif df_labelled['LABEL'][i] == "NOT_COGNITIVE_EFFORT":
            not_effort_dict[df_labelled[col_name][i]] += 1

    df_label_statistics['N_COGNITIVE_EFFORT'] = effort_dict.values()
    df_label_statistics['N_NOT_COGNITIVE_EFFORT'] = not_effort_dict.values()

    return df_label_statistics
