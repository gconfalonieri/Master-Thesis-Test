import pandas as pd
import toml
import labelling.utilities

config = toml.load('config.toml')

df_questions_statistics = pd.read_csv('datasets/results/questions_statistics.csv')

av_t_f = labelling.utilities.get_av_t_f()


for i in range(1, 46):
    if i not in config['general']['excluded_users']:
        user_id = 'USER_' + str(i)
        t_i_f = labelling.utilities.get_t_i_f(user_id)
        d_i_f = (t_i_f -av_t_f) / t_i_f
        print(d_i_f)