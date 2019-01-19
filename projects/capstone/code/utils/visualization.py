import math

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def distplot_all_cols(*dfs, num_cols=4, bins=None, columns=None):
    sample_df = dfs[0]

    if columns is None:
        columns = sample_df.columns

    num_rows = math.ceil(sample_df.shape[1] / num_cols)
    plt.rcParams['figure.figsize'] = [15, num_rows * 3.5]

    gs = gridspec.GridSpec(num_rows, num_cols)

    for idx, col_name in enumerate(columns):
        row = idx // num_cols
        col = idx % num_cols
        ax = plt.subplot(gs[row, col])
        for df in dfs:
            g = sns.distplot(df[col_name], ax=ax, bins=bins)
            g.set(xticklabels=[], yticklabels=[])

    plt.show()


def prediction_scatterplot():
    df = pd.read_csv('../../pred.csv')
    sns.scatterplot('y', 'pred', data=df)


if __name__ == '__main__':
    prediction_scatterplot()
