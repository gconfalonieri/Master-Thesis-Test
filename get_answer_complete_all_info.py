import pandas as pd
import statistics

df_answer_complete_all_info = pd.DataFrame(columns=['MEDIA_NAME','USER_ID','SPECIFIC_CATEGORY','MACRO_CATEGORY','CORRECT_ANSWER','USER_ANSWER','ANSWER_TIME'])

media_name = []
user_id_col = []
specific_category = []
macro_category = []
correct_answer = []
user_answer = []
answer_time = []

number_files = statistics.utilities.get_n_testers()

for i in range(1, number_files):

    if not(i==25 or i==26 or i==30):

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

df_answer_complete_all_info.to_csv('datasets/results/answers_complete_all_info.csv', index=False)