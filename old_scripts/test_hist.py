from matplotlib import pyplot as plt
import models.deep_learning_models
from sklearn.model_selection import train_test_split
import numpy as np
from keras import backend as K

round_values = []

complete_x_list = models.utilities.get_questions_oversampled_array()
complete_y_list = models.utilities.get_labels_questions_array()

X_train, X_test, y_train, y_test = train_test_split(complete_x_list, complete_y_list, test_size=0.2, shuffle=True)

X_train = np.array(X_train).astype(np.float32)
X_test = np.asarray(X_test).astype(np.float32)


# model = models.deep_learning_models.get_model_cnn1d_lstm(1, models.utilities.get_max_series_len(), 'relu', 'relu', 8, 4, 2, 64,  'mse', 'adam', 1, 0.2)
# model = models.deep_learning_models.get_model_cnn1d_lstm(1, models.utilities.get_max_series_len(), 'relu', 'relu', 8, 4, 2, 32,  'mse', 'adam', 1, 0.1)
model = models.deep_learning_models.get_model_cnn1d_lstm(1, models.utilities.get_max_series_len(), 'relu', 'relu', 8, 4, 2, 32,  'mse', 'adam', 1, 0.2)
# model = models.deep_learning_models.get_model_lstm(1, models.utilities.get_max_series_len(), 'relu', 'relu', 64,  'mse', 'adam', 1, 0.2)
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), shuffle=True)


get_relu_output = K.function(model.input, [model.layers[-1].output])

for x in get_relu_output([complete_x_list]):
    for y in x:
        round_values.append(round(y[0], 2))

history_dict = history.history

acc = history_dict["accuracy"][-1]
val_acc = history_dict["val_accuracy"][-1]
loss_values = history_dict['loss'][-1]
val_loss_values = history_dict['val_loss'][-1]
plt.title('ACC:( ' + str(round(acc,2)) + ' , ' + str(round(val_acc,2)) + ') - VAL:( ' + str(round(loss_values,2)) + ' , ' + str(round(val_loss_values,2)) + ')')
plt.hist(round_values)
plt.savefig('hist_cnn1d_lstm_3.png')
plt.clf()

