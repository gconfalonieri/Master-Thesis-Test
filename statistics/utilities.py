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
