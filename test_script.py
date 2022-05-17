import toml

import pandas as pd
from keras import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
import numpy as np

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

complete_x_list = np.array(complete_x_list)
complete_y_list = np.array(df_labelled['LABEL'])
X_train, X_test, y_train, y_test = train_test_split(complete_x_list, complete_y_list, test_size=0.2)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

X_train = X_train.reshape(-1, 1, X_train.shape[1])

model = Sequential()
model.add(LSTM(100, return_sequences=True))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=3, batch_size=64)
print(model.summary())