import pandas as pd
from pactools.grid_search import GridSearchCVProgressBar
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, make_scorer

from utils import preprocessing

RANDOM_STATE = 1
DATASET_FILE = 'data/pubg_dataset.csv'


def main():
    data = load_data()
    data = preprocess_data(data)

    X = data.drop('winPlacePerc', axis=1)
    y = data['winPlacePerc']

    regressor = RandomForestRegressor(random_state=RANDOM_STATE)
    # params = {
    #     'n_estimators': [1, 2, 4, 8, 16, 32, 64, 128],
    #     'criterion': ['mse', 'mae'],
    #     'max_depth': [None, 10, 100, 1000],
    #     'min_samples_split': [2, 4, 8, 16],
    #     'min_samples_leaf': [1, 2, 4, 8, 16],
    #     'max_features': ['auto', 'sqrt', 'log2'],
    #
    # }
    params = {
        'n_estimators': [16, 32],
        'max_depth': [None, 10],
    }

    scorer = make_scorer(mean_absolute_error)
    grid = GridSearchCVProgressBar(regressor, params, scoring=scorer, cv=5, n_jobs=-1)
    grid = grid.fit(X, y)
    reg = grid.best_estimator_

    print(f'Best parameters are {reg.get_params()}')

    return reg


def load_data():
    data = pd.read_csv(DATASET_FILE)
    return data


def preprocess_data(data):
    data = preprocessing.remove_outliers(data)
    data = preprocessing.scale_skewed_features(data)
    data = preprocessing.min_max_scale(data)
    return data


if __name__ == '__main__':
    main()
