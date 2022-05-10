import pandas as pd
import toml
import labelling.utilities

config = toml.load('config.toml')

for i in range(1, 53):

    if i not in config['general']['excluded_users']:

        user_id = 'USER_' + str(i)
        eye_source = 'datasets/eye-tracker/User ' + str(i) + '_all_gaze.csv'

        df_eye_complete = pd.read_csv(eye_source)
        df_eye = df_eye_complete.drop_duplicates('MEDIA_NAME', keep='last')

        for x in df_eye['MEDIA_NAME']:
            d_ij = labelling.utilities.get_d_ij(user_id, x)

        print(labelling.utilities.get_av_d_i(user_id))
