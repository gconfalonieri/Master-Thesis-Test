import os
import toml

config = toml.load('config.toml')


def get_n_testers():
    return len(os.listdir(config['path']['brainwaves_folder']))

