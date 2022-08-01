import toml

from models.utilities import get_questions_arrays_shifted, get_users_arrays_shifted, get_max_series_len_shifted, \
    get_questions_oversampled_validation_shifted

config = toml.load('config.toml')

total_arr = get_questions_oversampled_validation_shifted()

print(total_arr.shape)
# print(total_arr[0].shape)
# print(total_arr[1].shape)
