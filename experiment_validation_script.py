from sklearn.model_selection import train_test_split
import models.deep_learning_models as dl_models
import tensorflow as tf
import numpy as np

def train_model():
    complete_x_list = np.load('datasets/arrays/undersampled_shifted/input_1_1.npy', allow_pickle=True)
    complete_y_list = np.load('datasets/arrays/labels/labels_v2.npy', allow_pickle=True)
    X_train, X_test, y_train, y_test = train_test_split(complete_x_list, complete_y_list, test_size=0.2, shuffle=True)
    X_train = np.array(X_train).astype(np.float32)
    X_test = np.asarray(X_test).astype(np.float32)
    model = dl_models.get_model_lstm(0, 0, '', 'relu', 32, 'mse', 'adam', 1, 0.1)
    history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), shuffle=True)
    model.save('tf_models/test_model')

def evaluate_model():
    new_model = tf.keras.models.load_model('tf_models/test_model')
    complete_x_validation = np.load('datasets/arrays/undersampled_shifted/input_1_1_validation.npy', allow_pickle=True)
    complete_y_validation = np.load('datasets/arrays/labels/labels_validaton_v2.npy', allow_pickle=True)
    X_train = np.array(complete_x_validation).astype(np.float32)
    loss, acc = new_model.evaluate(X_train, complete_y_validation)
    print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))


train_model()