import toml
import pandas as pd

config = toml.load('config.toml')

complete_x_list = []
complete_y_list = []

for i in range(1, 53):
    user_id = 'USER_' + str(i)
    if i not in config['general']['excluded_users']:
        path = 'datasets/sync_datasets/sync_dataset_' + user_id.lower() + '.csv'
        df_sync = pd.read_csv(path)
        media_names = df_sync.drop_duplicates('MEDIA_ID', keep='last')['MEDIA_ID']
        for name in media_names:
            reduced_df = df_sync[df_sync['MEDIA_ID'] == name]
            user_list = reduced_df['beta1'].tolist()
            complete_x_list.append(user_list)

path_labelled_df = config['path']['labelled_dataset']
df_labelled = pd.read_csv(path_labelled_df)
complete_y_list = df_labelled['LABEL'].tolist()