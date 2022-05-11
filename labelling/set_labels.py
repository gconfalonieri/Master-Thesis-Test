import pandas as pd
import statistics.utilities
import toml

config = toml.load('config.toml')

pd.options.mode.chained_assignment = None

def set_label_general_with_threshold(df_complete, col_name, threshold_dict):

    labels = []

    min_times_dict = statistics.utilities.get_min_right_dict(df_complete, 'MEDIA_NAME')

    for i in df_complete.index:
        is_right_answer = statistics.utilities.get_answer_right_or_wrong(df_complete['MEDIA_NAME'][i], df_complete['USER_ANSWER'][i], df_complete['CORRECT_ANSWER'][i])
        answer_time = df_complete['T_I_J'][i]
        current_min_time = min_times_dict[df_complete[col_name][i]]
        current_threshold = threshold_dict[df_complete[col_name][i]]
        if not is_right_answer and answer_time < current_min_time:
            labels.append('0')
        elif not is_right_answer and answer_time >= current_min_time:
            labels.append('1')
        elif is_right_answer and answer_time > current_threshold:
            labels.append('1')
        elif is_right_answer and current_min_time <= answer_time <= current_threshold:
            labels.append('0')

    return labels

def set_label_times_only_v1(df_complete):

    labels = []

    for i in df_complete.index:
        n1 = df_complete['T_I_J'][i]
        n2 = df_complete['AVERAGE_T_J'][i] * (1 + df_complete['AVERAGE_D_I'][i])
        if n1 > n2:
            labels.append('1')
        else:
            labels.append('0')

    return labels


def set_label_times_only_v2(df_complete):

    labels = []

    for i in df_complete.index:
        n1 = df_complete['T_I_J'][i] * (1 - df_complete['AVERAGE_D_I'][i])
        n2 = df_complete['AVERAGE_T_J'][i]
        if n1 > n2:
            labels.append('1')
        else:
            labels.append('0')

    return labels


def get_df_answers_labelled(df_complete, labelling_type):

    df_complete_labelled = df_complete
    labels = []

    if labelling_type == 'GENERAL_AVERAGE_MEDIA_NAME':
        mean_answer_dict = statistics.utilities.get_mean_dict(statistics.utilities.get_times_dict(df_complete_labelled, 'MEDIA_NAME'))
        labels = set_label_general_with_threshold(df_complete_labelled, 'MEDIA_NAME', mean_answer_dict)
    elif labelling_type == 'TIMES_ONLY_V1':
        labels = set_label_times_only_v1(df_complete)
    elif labelling_type == 'TIMES_ONLY_V2':
        labels = set_label_times_only_v2(df_complete)

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
        if df_labelled['LABEL'][i] == "1":
            effort_dict[df_labelled[col_name][i]] += 1
        elif df_labelled['LABEL'][i] == "0":
            not_effort_dict[df_labelled[col_name][i]] += 1

    df_label_statistics['N_COGNITIVE_EFFORT'] = effort_dict.values()
    df_label_statistics['N_NOT_COGNITIVE_EFFORT'] = not_effort_dict.values()

    return df_label_statistics
