from collections import Counter

import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

SKEWED = ['boosts', 'damageDealt', 'headshotKills', 'heals', 'kills', 'killStreaks', 'longestKill',
          'rideDistance', 'roadKills', 'swimDistance', 'vehicleDestroys', 'walkDistance',
          'weaponsAcquired', 'winPoints']


def remove_outliers(df):
    new_df = df.copy().reset_index(drop=True)

    outliers_counter = Counter()

    # For each feature find the data points with extreme high or low values
    for col in new_df.columns[:-1]:
        # Calculate Q1 and Q3 for the given feature
        Q1 = np.percentile(new_df[col], 25)
        Q3 = np.percentile(new_df[col], 75)

        # Calculate the outlier step
        step = 1.5 * (Q3 - Q1)

        # Calculate outliers
        outliers_table = new_df[(new_df[col] < Q1 - step) | (new_df[col] > Q3 + step)]
        outliers_counter.update(outliers_table.index)

    # Select the indices of the outliers
    outliers = [idx for idx, count in outliers_counter.items() if count >= 5]

    # Remove the outliers
    new_df.drop(new_df.index[outliers], inplace=True)
    new_df.reset_index(drop=True)

    return new_df


def scale_skewed_features(df):
    new_df = df.copy()

    # Apply Box-Cox to the skewed features
    new_df[SKEWED] = new_df[SKEWED].apply(lambda x: stats.boxcox(x + 1)[0])

    return new_df


def min_max_scale(df):
    new_df = df.copy()

    # Don't scale the winPlacePerc
    columns = new_df.columns[:-1]

    # Scale all features
    scaler = MinMaxScaler()
    new_df[columns] = scaler.fit_transform(new_df[columns])

    return new_df
