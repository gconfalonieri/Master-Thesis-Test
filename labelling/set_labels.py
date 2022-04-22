import pandas as pd
import statistics.utilities
import toml

config = toml.load('config.toml')

pd.options.mode.chained_assignment = None


def get_answer_complete_all_info_df(n_users):

    df_answer_complete_all_info = pd.DataFrame(columns=['MEDIA_NAME', 'USER_ID', 'SPECIFIC_CATEGORY', 'MACRO_CATEGORY', 'CORRECT_ANSWER', 'USER_ANSWER', 'ANSWER_TIME'])

    media_name = []
    user_id_col = []
    specific_category = []
    macro_category = []
    correct_answer = []
    user_answer = []
    answer_time = []

    for i in range(1, n_users):

        if i not in config['general']['excluded_users']:

            user_id = 'USER_' + str(i)
            eye_source = 'datasets/eye-tracker/User ' + str(i) + '_all_gaze.csv'
            user_answers_source = 'datasets/users/answers/answer_user_' + str(i) + '.csv'

            df_user_answer = pd.read_csv(user_answers_source)
            df_categories_complete = pd.read_csv('datasets/questions/categories_complete.csv')

            df_eye_complete = pd.read_csv(eye_source)
            df_eye = df_eye_complete.drop_duplicates('MEDIA_NAME', keep='last')

            for x in df_eye['MEDIA_NAME']:
                media_name.append(x)
                user_id_col.append(user_id)
                specific_category.append(statistics.utilities.get_specific_category(x, df_categories_complete))
                macro_category.append(statistics.utilities.get_macro_category(x, df_categories_complete))
                correct_answer.append(statistics.utilities.get_correct_answer(x, df_user_answer))
                user_answer.append(statistics.utilities.get_user_answer(x, df_user_answer))
                answer_time.append(statistics.utilities.get_answer_time(x, df_eye_complete))

            print(user_id + " COMPLETE")

    df_answer_complete_all_info['MEDIA_NAME'] = media_name
    df_answer_complete_all_info['USER_ID'] = user_id_col
    df_answer_complete_all_info['SPECIFIC_CATEGORY'] = specific_category
    df_answer_complete_all_info['MACRO_CATEGORY'] = macro_category
    df_answer_complete_all_info['CORRECT_ANSWER'] = correct_answer
    df_answer_complete_all_info['USER_ANSWER'] = user_answer
    df_answer_complete_all_info['ANSWER_TIME'] = answer_time

    return df_answer_complete_all_info


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
