import numpy as np
from sklearn.model_selection import train_test_split

import models

complete_x_list = models.utilities.get_questions_padded_array()
np.save('datasets/arrays/undersampled/input_0_1_padded_begin.npy', complete_x_list)
