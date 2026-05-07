import numpy as np
from sklearn.metrics import confusion_matrix

def amae(y_true: np.ndarray, y_pred: np.ndarray):
    """Copied from dlordinal package: https://github.com/ayrna/dlordinal
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    n_class = cm.shape[0]
    costs = np.reshape(np.tile(range(n_class), n_class), (n_class, n_class))
    costs = np.abs(costs - np.transpose(costs))
    errors = costs * cm

    # Remove rows with all zeros in the confusion matrix
    non_zero_cm_rows = ~np.all(cm == 0, axis=1)
    errors = errors[non_zero_cm_rows]
    cm = cm[non_zero_cm_rows]

    per_class_maes = np.sum(errors, axis=1) / np.sum(cm, axis=1).astype("double")
    return np.mean(per_class_maes)