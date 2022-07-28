from matplotlib import pyplot as plt
import models.deep_learning_models
from sklearn.model_selection import train_test_split
import numpy as np
from keras import backend as K

loss_list = []
accuracy_list = []

complete_x_list = models.utilities.get_questions_oversampled_array()
complete_y_list = models.utilities.get_labels_questions_array()
# np.save('interpolation_x_1d.npy', complete_x_list)
# np.save('datasets/arrays/labels/labels_v2.npy', complete_y_list)

# complete_x_list = np.load('datasets/arrays/undersampled_shifted/input_1_1_oversampled.npy', allow_pickle=True)
# complete_y_list = np.load('datasets/arrays/labels/labels_v2.npy', allow_pickle=True)

# complete_x_list = np.expand_dims(complete_x_list, 2)
# complete_y_list = np.expand_dims(complete_y_list, 2)

print("# TRAIN SERIES #")
print(complete_x_list.shape)

print("# TRAIN LABELS #")
print(complete_y_list.shape)

# X_train, X_test, y_train, y_test = train_test_split(np.ones((46 * 24, 2, 100)), np.ones((46 * 24, 1)), test_size=0.2)
X_train, X_test, y_train, y_test = train_test_split(complete_x_list, complete_y_list, test_size=0.2, shuffle=True)

X_train = np.array(X_train).astype(np.float32)
X_test = np.asarray(X_test).astype(np.float32)

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

# 6,LSTM,TIMES_ONLY_V2,-1_1_OVERSAMPLED,mean_squared_error,adam,0,,,relu,64,,,,1,0.2
# 7,LSTM,TIMES_ONLY_V2,-1_1_OVERSAMPLED,mean_squared_error,adam,1,2646,relu,relu,64,,,,1,0.2

model = models.deep_learning_models.get_model_cnn1d_lstm(1, models.utilities.get_max_series_len(), 'relu', 'relu', 8, 4, 2, 64,  'mse', 'adam', 1, 0.2)
name = 'LSTM - 6'
history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), shuffle=True)

get_relu_output = K.function(model.input, [model.layers[-1].output])
print(get_relu_output([complete_x_list]))

round_values = []
for x in get_relu_output([complete_x_list]):
    for y in x:
        round_values.append(round(y[0], 2))


# print(round_values)

plt.hist(round_values)
plt.show()

model.summary()

history_dict = history.history

# models.plots.plot_model_loss(history_dict, name)
# plt.clf()
# models.plots.plot_model_accuracy(history_dict, name)

# results = model.evaluate(X_test, y_test)

# print(results)

# model.save('model_results/padding/cnn2d')
