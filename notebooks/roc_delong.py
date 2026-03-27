import numpy as np
from scipy import stats

def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2

def delong_roc_variance(ground_truth, predictions):
    order = np.argsort(-predictions)
    predictions = predictions[order]
    ground_truth = ground_truth[order]

    distinct_value_indices = np.where(np.diff(predictions))[0]
    threshold_idxs = np.r_[distinct_value_indices, ground_truth.size - 1]

    tpr = np.cumsum(ground_truth)[threshold_idxs] / ground_truth.sum()
    fpr = (1 + threshold_idxs - np.cumsum(ground_truth)[threshold_idxs]) / (ground_truth.size - ground_truth.sum())

    auc = np.trapz(tpr, fpr)

    m = ground_truth.sum()
    n = ground_truth.size - m
    v01 = (tpr * (1 - tpr)) / m
    v10 = (fpr * (1 - fpr)) / n
    se_auc = np.sqrt(v01.sum() + v10.sum())
    return auc, se_auc

def delong_roc_test(y_true, y_scores1, y_scores2):
    auc1, se1 = delong_roc_variance(y_true, y_scores1)
    auc2, se2 = delong_roc_variance(y_true, y_scores2)
    diff = auc1 - auc2
    se_diff = np.sqrt(se1**2 + se2**2)
    z = diff / se_diff
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return p_value
