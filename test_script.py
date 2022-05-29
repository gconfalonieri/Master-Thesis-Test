import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.utils.np_utils import to_categorical
import tensorflow as tf
import keras.preprocessing
import pandas as pd
from keras import Sequential, Model
from keras.layers import LSTM, Dense, Input, Embedding, Conv1D, Conv2D, MaxPooling1D
from sklearn.model_selection import train_test_split
import numpy as np
import toml


config = toml.load('config.toml')

features = ['beta1', 'beta2']

complete_x_list = []
complete_y_list = []

users = []


for i in range(1, 53):
    user_id = 'USER_' + str(i)
    if i not in config['general']['excluded_users']:
        user_list = []
        users.append(user_id)
        path = 'datasets/sync_datasets/sync_dataset_' + user_id.lower() + '.csv'
        df_sync = pd.read_csv(path)
        media_names = df_sync.drop_duplicates('media_name', keep='last')['media_name']
        for name in media_names:
            reduced_df = df_sync[df_sync['media_name'] == name]
            question_list = []
            for f in features:
                question_list.append(np.asarray(reduced_df[f]))
            user_list.append(question_list)
        complete_x_list.append(user_list)

path_labelled_df = config['path']['labelled_dataset']
df_labelled = pd.read_csv(path_labelled_df)

c = 0
for i in range(1, 53):
    if i not in config['general']['excluded_users']:
        user_list = []
        for j in range(1, 25):
            user_list.append(df_labelled['LABEL'][c])
            c += 1
        complete_y_list.append(user_list)

complete_x_list = np.array(complete_x_list, dtype=np.ndarray)
complete_y_list = np.asarray(complete_y_list).astype('float32')

print("# TRAIN SERIES #")
print(complete_x_list.shape)
print(type(complete_x_list))
print(complete_x_list[0].shape)
print(type(complete_x_list[0]))
print(complete_x_list[0][0].shape)
print(type(complete_x_list[0][0]))
print(complete_x_list[0][0][0].shape)
print(type(complete_x_list[0][0][0]))
print(type(complete_x_list[0][0][0][0]))

print("# TRAIN LABELS #")
print(complete_y_list.shape)
print(type(complete_y_list))
print(complete_y_list[0].shape)
print(type(complete_y_list[0]))
print(complete_y_list[0][0].shape)
print(type(complete_y_list[0][0]))

X_train, X_test, y_train, y_test = train_test_split(complete_x_list, complete_y_list, test_size=0.2)

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

model = Sequential()
model.add(Conv1D(filters=256, kernel_size=2))
model.add(MaxPooling1D(pool_size=2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train)

# input_tensor = Input(shape=(36, config['general']['n_questions'], 1))
# output_layer = Dense(1)
# output_tensor = output_layer(input_tensor)
#
# model = Model(input_tensor, output_tensor, name='Model')
# model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
# model.fit(X_train, y_train, epochs=30, batch_size=32)
# print(model.summary())

# model.add(Input(shape=(len(users), config['general']['n_questions'], 1, )))
# model.add(Input())
# model.add(LSTM(32))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
# model.fit(X_train, y_train, epochs=30, batch_size=32)
# print(model.summary())
