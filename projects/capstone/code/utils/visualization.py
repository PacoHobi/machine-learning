import math

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn as sns


def distplot_all_cols(*dfs, num_cols=4, bins=None, columns=None):
    """Distribution plots of all features in a 4 by 4 grid."""
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


def boxplot_all_cols(*dfs, num_cols=4, columns=None):
    """Box plots of all features in a 4 by 4 grid."""
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
            g = sns.boxplot(df[col_name], ax=ax)
            g.set(xticklabels=[], yticklabels=[])

    plt.show()
