import os
import toml

config = toml.load('config.toml')


def get_n_testers():
    return len(os.listdir(config['path']['brainwaves_folder']))


def get_index_column(n):
    index = []
    for i in range(1, n+1):
        index.append(i)
    return index


def get_media_name_column(df_reference):
    media_name = []
    for x in df_reference['MEDIA_NAME']:
        media_name.append(x)
    return media_name


def correct_answer_column(df_question, df_correct_answer):
    correct_answer_column = []
    for x in df_question['MEDIA_NAME']:
        for i in df_correct_answer.index:
            if x == df_correct_answer['MEDIA_NAME'][i]:
                correct_answer_column.append(df_correct_answer['CORRECT_ANSWER'][i])
                break
    return correct_answer_column


def get_correct_answer(target_element, reference_df):
    for i in reference_df.index:
        if target_element == reference_df['MEDIA_NAME'][i]:
            return reference_df['CORRECT_ANSWER'][i]


def get_specific_category(target_element, reference_df):
    for i in reference_df.index:
        if target_element == reference_df['MEDIA_NAME'][i]:
            return reference_df['SPECIFIC_CATEGORY'][i]


def get_macro_category(target_element, reference_df):
    for i in reference_df.index:
        if target_element == reference_df['MEDIA_NAME'][i]:
            return reference_df['MACRO_CATEGORY'][i]


def get_user_answer(target_element, reference_df):
    for i in reference_df.index:
        if target_element == reference_df['MEDIA_NAME'][i]:
            return reference_df['USER_ANSWER'][i]


def get_answer_time(target_element, reference_df):
    df_media = reference_df[reference_df['MEDIA_NAME'] == target_element]
    return df_media.iloc[-1, 3]
