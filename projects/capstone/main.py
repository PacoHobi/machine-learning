from time import time

import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from utils import preprocessing

RANDOM_STATE = 1
DATASET_FILE = 'data/pubg_dataset.csv'


def main():
    data = load_data()

    configurations = [
        {
            'steps': [
            ],
            'regressor_class': GradientBoostingRegressor
        },
        {
            'steps': [
                preprocessing.remove_outliers,
            ],
            'regressor_class': GradientBoostingRegressor
        },
        {
            'steps': [
                preprocessing.remove_outliers,
                preprocessing.scale_skewed_features,
            ],
            'regressor_class': GradientBoostingRegressor
        },
        {
            'steps': [
                preprocessing.remove_outliers,
                preprocessing.scale_skewed_features,
                preprocessing.min_max_scale,
            ],
            'regressor_class': GradientBoostingRegressor
        },
        {
            'steps': [
                preprocessing.scale_skewed_features,
                preprocessing.min_max_scale,
            ],
            'regressor_class': GradientBoostingRegressor
        },
        {
            'steps': [
                preprocessing.min_max_scale,
            ],
            'regressor_class': GradientBoostingRegressor
        },
        {
            'steps': [
                preprocessing.scale_skewed_features,
            ],
            'regressor_class': GradientBoostingRegressor
        }
    ]

    for configuration in configurations:
        run_configuration(data, configuration)


def load_data():
    df = pd.read_csv(DATASET_FILE)
    return df


def run_configuration(data, configuration):
    # Run steps
    steps = configuration['steps']
    for step in steps:
        data = step(data)

    # Separate features
    features = data.drop('winPlacePerc', axis=1)
    win_place_perc = data['winPlacePerc']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, win_place_perc, test_size=0.2, random_state=RANDOM_STATE
    )

    # Initialize classifier
    regressor_class = configuration['regressor_class']
    regressor = regressor_class(random_state=RANDOM_STATE)

    results = train_predict(regressor, X_train, y_train, X_test, y_test)

    print(
        '--------------------------------------\n'
        f'Regressor class: {regressor_class.__name__}\n'
        f'Preprocessing:   {", ".join(step.__name__ for step in steps)}\n'
        f'Train time:      {results["train_time"]:f}\n'
        f'Prediction time: {results["pred_time"]:f}\n'
        f'MAE train:       {results["mae_train"]:.4f}\n'
        f'MAE test:        {results["mae_test"]:.4f}'
    )


def train_predict(learner, X_train, y_train, X_test, y_test):
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''

    results = {}

    # Fit the learner
    start = time()
    learner = learner.fit(X_train, y_train)
    end = time()

    # Calculate the training time
    results['train_time'] = end - start

    # Predict for the testing and training sets
    start = time()  # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train)
    end = time()  # Get end time

    # Calculate the total prediction time
    results['pred_time'] = end - start

    # Compute MAE for the training set
    results['mae_train'] = mean_absolute_error(y_train, predictions_train)

    # Compute MAE for the testing set
    results['mae_test'] = mean_absolute_error(y_test, predictions_test)

    # Return the results
    return results


if __name__ == '__main__':
    main()
