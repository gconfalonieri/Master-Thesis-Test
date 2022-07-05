import numpy as np
from sklearn.model_selection import train_test_split

import models

complete_x_list = models.utilities.get_questions_padded_array()
np.save('datasets/arrays/undersampled/input_0_1_padded_end.npy', complete_x_list)
complete_x_list = models.utilities.get_questions_oversampled_array()
np.save('datasets/arrays/undersampled/input_0_1_oversampled.npy', complete_x_list)
