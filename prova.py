from models import utilities

over = utilities.get_questions_oversampled_array()
max_len = utilities.get_max_series_len()

print(over)

for x in over:
    print(len(x[0]))

print(max_len)
