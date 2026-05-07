from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve


def compute_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else np.nan


def safe_auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_prob)


def safe_ap(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return average_precision_score(y_true, y_prob)


def compute_metrics(y_true, y_prob, threshold):
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    auc = safe_auc(y_true, y_prob)
    ap = safe_ap(y_true, y_prob)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    spec = compute_specificity(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "AUC": auc,
        "PR_AUC(AP)": ap,
        "F1": f1,
        "Recall(Sensitivity)": rec,
        "Specificity": spec,
        "Accuracy": acc,
        "TN": int(cm[0, 0]),
        "FP": int(cm[0, 1]),
        "FN": int(cm[1, 0]),
        "TP": int(cm[1, 1]),
        "Threshold": float(threshold),
    }, cm


def calibration_bins(y_true, y_prob, n_bins=10):
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    return frac_pos, mean_pred, brier_score_loss(y_true, y_prob)
