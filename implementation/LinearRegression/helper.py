import os

def get_redwine_dir():
    dirname = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    datasets = os.path.join(dirname, 'datasets')
    redwine_dir = os.path.join(datasets, 'redwine')
    return redwine_dir

def get_dataset_path():
    redwine_dir = get_redwine_dir()
    redwine = os.path.join(redwine_dir, 'winequality-red.csv')
    return redwine

def get_debug_path():
    redwine_dir = get_redwine_dir()
    proj_debug = os.path.join(redwine_dir, 'debug.txt')
    return proj_debug