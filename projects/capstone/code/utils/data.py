import pandas as pd

import config
from utils import preprocessing


def load_data():
    data = pd.read_csv(config.DATASET_FILE)
    return data


def preprocess_data(data):
    data = preprocessing.remove_outliers(data)
    data = preprocessing.scale_skewed_features(data)
    data = preprocessing.min_max_scale(data)
    return data
