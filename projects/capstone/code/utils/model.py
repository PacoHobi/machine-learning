from time import time

from sklearn.metrics import mean_absolute_error


def train_predict(learner, X_train, y_train, X_test, y_test):
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
