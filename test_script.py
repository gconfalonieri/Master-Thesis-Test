import matplotlib
matplotlib.use('Agg')
import models.deep_learning_models
import numpy as np

# complete_x_list = models.utilities.get_questions_oversampled_array()
# np.save('datasets/arrays/undersampled_shifted/input_1_1_oversampled.npy', complete_x_list)
# complete_x_list = np.load('datasets/arrays/undersampled_shifted/input_1_1_oversampled.npy', allow_pickle=True)
# complete_y_list = models.utilities.get_labels_questions_array()
# np.save('datasets/arrays/labels/labels_v2.npy', complete_y_list)
# print(complete_y_list.shape)

complete_x_list = models.utilities.get_questions_oversampled_array_shifted()
np.save('datasets/arrays/undersampled_shifted/input_1_1.npy', complete_x_list)
complete_y_list = models.utilities.get_labels_questions_array_shifted()
np.save('datasets/arrays/labels/labels_v2.npy', complete_y_list)

complete_x_validation = models.utilities.get_questions_oversampled_validation_shifted()
np.save('datasets/arrays/undersampled_shifted/input_1_1_validation.npy', complete_x_validation)
complete_y_validation = models.utilities.get_labels_questions_validation_shifted()
np.save('datasets/arrays/labels/labels_validaton_v2.npy', complete_y_validation)