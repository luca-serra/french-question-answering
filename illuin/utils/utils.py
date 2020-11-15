import numpy as np


def build_features(row):
    """
    Given a row of the TF-IDF matrix, return (mean, ratio_complete) of the row
    'ratio_complete' is the ratio of non-zero terms in the row.
    """
    count = 0
    for value in row:
        if value != 0:
            count += 1

    return [np.mean(row), count / row.shape[0]]
