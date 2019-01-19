from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.model_selection import RandomizedSearchCV

import config
from utils.data import load_data, preprocess_data


def random_grid_tuning():
    # Load and preprocess data
    data = load_data()
    data = preprocess_data(data)

    # Separate features
    X = data.drop('winPlacePerc', axis=1)
    y = data['winPlacePerc']

    # Init grid
    random_grid = {
        'n_estimators': list(range(20, 220, 20)),
        'max_depth': [None] + list(range(10, 110, 10)),
        'criterion': ['mse', 'mae'],
        'min_samples_leaf': [1, 2, 4, 8],
        'min_samples_split': [2, 4, 8],
        'max_features': ['auto', 'sqrt'],
    }
    regressor = RandomForestRegressor(random_state=config.RANDOM_STATE)
    scorer = make_scorer(mean_absolute_error)
    grid = RandomizedSearchCV(
        estimator=regressor,
        param_distributions=random_grid,
        scoring=scorer,
        n_iter=100,
        random_state=config.RANDOM_STATE,
        cv=3, verbose=50,  # n_jobs=-1
    )
    grid = grid.fit(X, y)

    print(f'Best parameters are {grid.best_params_}')

    return grid


if __name__ == '__main__':
    random_grid_tuning()
