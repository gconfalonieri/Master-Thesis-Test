import toml

from models.utilities import get_questions_arrays_shifted, get_users_arrays_shifted, get_max_series_len_shifted, \
    get_questions_oversampled_validation_shifted

config = toml.load('config.toml')

print(get_questions_oversampled_validation_shifted().shape)
