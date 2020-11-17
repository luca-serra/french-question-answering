import numpy as np


def build_features(tfidf_matrix):
    """Given the (m x n) tfidf matrix, return the (m x 2) feature matrix"""

    def _build_row_features(row):
        """
        Given a row of the TF-IDF matrix, return (mean, ratio_complete) of the row
        'ratio_complete' is the ratio of non-zero terms in the row.
        """
        count = 0
        for value in row:
            if value != 0:
                count += 1
        return [np.mean(row), count / row.shape[0]]

    features = []
    for row_idx in range(tfidf_matrix.shape[0]):
        features.append(_build_row_features(tfidf_matrix[row_idx, :]))

    return np.array(features)


def argmax_n(array, n):
    """Return the indices of the top n values of array"""
    indices = np.argpartition(array, -n)[-n:]
    indices = indices[np.argsort(-array[indices])]
    return indices
