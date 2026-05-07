from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def build_dca_thresholds(tmin=0.05, tmax=0.50, step=0.01):
    thresholds = np.arange(tmin, tmax + step, step, dtype=float)
    thresholds = thresholds[(thresholds > 0) & (thresholds < 1)]
    return np.round(thresholds, 6)


def net_benefit_model(y_true, y_prob, threshold):
    pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    n = len(y_true)
    odds = threshold / (1.0 - threshold)
    return float((tp / n) - (fp / n) * odds) if n else np.nan


def net_benefit_all(y_true, threshold):
    y_true = np.asarray(y_true)
    n = len(y_true)
    if n == 0:
        return np.nan
    event_rate = np.mean(y_true == 1)
    odds = threshold / (1.0 - threshold)
    return float(event_rate - (1.0 - event_rate) * odds)


def net_benefit_none(y_true, threshold):
    return 0.0


def decision_curve_df(y_true, y_prob, thresholds, model_name="model", dataset_name="test"):
    rows = []
    for thr in thresholds:
        rows.append({
            "dataset": dataset_name,
            "model": model_name,
            "threshold": float(thr),
            "net_benefit_model": net_benefit_model(y_true, y_prob, thr),
            "net_benefit_all": net_benefit_all(y_true, thr),
            "net_benefit_none": net_benefit_none(y_true, thr),
        })
    return pd.DataFrame(rows)
