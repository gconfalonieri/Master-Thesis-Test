from sklearn.model_selection import train_test_split
import models.deep_learning_models as dl_models
import models.plots
import tensorflow as tf
import numpy as np

def train_model():
    complete_x_list = np.load('datasets/arrays/undersampled_shifted/input_1_1.npy', allow_pickle=True)
    complete_y_list = np.load('datasets/arrays/labels/labels_v2.npy', allow_pickle=True)
    print(complete_x_list.shape)
    print(complete_y_list.shape)
    X_train, X_test, y_train, y_test = train_test_split(complete_x_list, complete_y_list, test_size=0.13, shuffle=True)
    X_train = np.array(X_train).astype(np.float32)
    X_test = np.asarray(X_test).astype(np.float32)
    model = dl_models.get_model_lstm(0, 0, '', 'relu', 32, 'mse', 'adam', 1, 0.1)
    history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), shuffle=True)
    history_dict = history.history
    name = 'BEST MODEL'
    models.plots.plot_model_loss(history_dict, name)
    models.plots.plot_model_accuracy(history_dict, name)
    model.save('tf_models/best_model_test')

def evaluate_model():
    new_model = tf.keras.models.load_model('tf_models/AAA_test_model_40')
    complete_x_validation = np.load('datasets/arrays/undersampled_shifted/input_1_1_validation.npy', allow_pickle=True)
    complete_y_validation = np.load('datasets/arrays/labels/labels_validaton_v2.npy', allow_pickle=True)
    X_train = np.array(complete_x_validation).astype(np.float32)
    loss, acc = new_model.evaluate(X_train, complete_y_validation)
    print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))


evaluate_model()