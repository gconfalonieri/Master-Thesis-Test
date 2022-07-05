import os

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import models.utilities
import models.deep_learning_models
from sklearn.model_selection import train_test_split
import numpy as np

f = open('results.csv', 'w')
f.write('index,model,label,input,loss_type,optimizer_type,dense_input,dense_input_dim,dense_input_activation,'
        'dense_output_activation,lstm_units,cnn_fiters,cnn_kernel,cnn_pool_size,dropout,dropout_value\n')
f.close()

def write_line(c, model, label_name, input_name, loss_type, optimizer_type, dense_input, dense_input_dim,
               dense_input_activation, dense_output_activation, lstm_cells, n_cnn_filters, cnn_kernel_size,
               cnn_pool_size, dropout, dropout_value):
    line = str(c) + ',' + model + ',' + label_name + ',' + input_name + ',' + loss_type + ',' \
           + optimizer_type + ',' + str(dense_input) + ',' + str(dense_input_dim) + ',' \
           + dense_input_activation + ',' + dense_output_activation + ',' + str(lstm_cells) \
           + ',' + str(n_cnn_filters) + ',' + str(cnn_kernel_size) + ',' + str(cnn_pool_size) + ',' \
           + str(dropout) + ',' + str(dropout_value) + '\n'
    f = open('results.csv', 'a')
    f.write(line)
    f.close()

loss_list = []
accuracy_list = []

# complete_x_list = numpy.load('datasets/numpy_arrays/all_windowed_array_data.npy', allow_pickle=True)
# complete_y_list = numpy.load('datasets/numpy_arrays/all_windowed_array_labels.npy', allow_pickle=True)

complete_x_list = models.utilities.get_questions_padded_array()
complete_y_list = models.utilities.get_labels_questions_array()
# np.save('interpolation_x_1d.npy', complete_x_list)
# np.save('labels_y.npy', complete_y_list)

# complete_x_list = np.load('padded_x_1d.npy', allow_pickle=True)
# complete_y_list = np.load('labels_y.npy', allow_pickle=True)

complete_x_list = np.expand_dims(complete_x_list, 2)
complete_y_list = np.expand_dims(complete_y_list, 2)

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



model = models.deep_learning_models.get_model_cnn2d(1, 1000, 'relu', 'linear', 2, 2, 10, 'mean_squared_error', 'adam', 1, 0.5)
name = 'CNN1D'

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

model.summary()

history_dict = history.history

print(history_dict['accuracy'][-1])

# models.plots.plot_model_loss(history_dict, name)
# plt.clf()
# models.plots.plot_model_accuracy(history_dict, name)

# results = model.evaluate(X_test, y_test)

# print(results)

# model.save('model_results/padding/cnn2d')