import pandas as pd

import statistics.utilities

pd.options.mode.chained_assignment = None

def get_df_answers_labelled(df_complete, min_times_dict, threshold_dict):

    df_complete_labelled = df_complete
    df_complete_labelled['LABEL'] = [None] * len(df_complete_labelled['MEDIA_NAME'])

    for i in df_complete.index:
        is_right_answer = statistics.utilities.get_answer_right_or_wrong(df_complete_labelled['USER_ANSWER'][i], df_complete_labelled['CORRECT_ANSWER'][i])
        answer_time = df_complete_labelled['ANSWER_TIME'][i]
        current_min_time = min_times_dict[df_complete_labelled['MEDIA_NAME'][i]]
        current_threshold = threshold_dict[df_complete_labelled['MEDIA_NAME'][i]]
        if not is_right_answer:
            df_complete_labelled['LABEL'][i] = 'COGNITIVE_EFFORT'
        elif is_right_answer and answer_time > current_threshold:
            df_complete_labelled['LABEL'][i] = 'COGNITIVE_EFFORT'
        elif is_right_answer and current_min_time < answer_time < current_threshold:
            df_complete_labelled['LABEL'][i] = 'NOT_COGNITIVE_EFFORT'

    return df_complete_labelled
