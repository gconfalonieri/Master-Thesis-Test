import numpy as np
from sklearn.model_selection import train_test_split

import models

complete_x_list = models.utilities.get_questions_padded_array()
complete_y_list = models.utilities.get_labels_questions_array()

model = models.deep_learning_models.get_model_lstm(2, 'linear', 'categorical_crossentropy', 'SGD')

X_train, X_test, y_train, y_test = train_test_split(complete_x_list, complete_y_list, test_size=0.2, shuffle=True)

X_train = np.array(X_train).astype(np.float32)
X_test = np.asarray(X_test).astype(np.float32)

history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

model.summary()
