from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve

from utils.metrics import compute_specificity


def youden_threshold(y_true, y_prob):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    idx = int(np.argmax(j))
    return float(thr[idx]), float(tpr[idx]), float(1 - fpr[idx])


def threshold_for_target_sensitivity(y_true, y_prob, target_sens=0.90):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    spec = 1 - fpr
    valid = np.where(tpr >= target_sens)[0]
    if len(valid) == 0:
        thr0, sens0, spec0 = youden_threshold(y_true, y_prob)
        return float(thr0), float(sens0), float(spec0), True
    idx = valid[np.argmax(spec[valid])]
    return float(thr[idx]), float(tpr[idx]), float(spec[idx]), False


def threshold_for_target_specificity(y_true, y_prob, target_spec=0.90):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    spec = 1 - fpr
    valid = np.where(spec >= target_spec)[0]
    if len(valid) == 0:
        thr0, sens0, spec0 = youden_threshold(y_true, y_prob)
        return float(thr0), float(sens0), float(spec0), True
    idx = valid[np.argmax(tpr[valid])]
    return float(thr[idx]), float(tpr[idx]), float(spec[idx]), False


def threshold_max_f1(y_true, y_prob):
    from sklearn.metrics import f1_score, recall_score
    thresholds = np.unique(y_prob)
    best_thr, best_f1, best_sens, best_spec = 0.5, -1, np.nan, np.nan
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        sens = recall_score(y_true, y_pred, zero_division=0)
        spec = compute_specificity(y_true, y_pred)
        if f1 > best_f1:
            best_f1, best_thr, best_sens, best_spec = f1, thr, sens, spec
    return float(best_thr), float(best_sens), float(best_spec)


def threshold_fixed(y_true, y_prob, fixed_threshold=0.5):
    from sklearn.metrics import recall_score
    y_pred = (y_prob >= fixed_threshold).astype(int)
    sens = recall_score(y_true, y_pred, zero_division=0)
    spec = compute_specificity(y_true, y_pred)
    return float(fixed_threshold), float(sens), float(spec)


def build_threshold_jobs(sensitivity_targets, specificity_targets, fixed_threshold):
    jobs = [{"policy_group": "youden", "target_value": np.nan, "policy_label": "youden"},
            {"policy_group": "fixed", "target_value": fixed_threshold, "policy_label": f"fixed_{fixed_threshold:.2f}"}]
    for t in sensitivity_targets:
        jobs.append({"policy_group": "target_sensitivity", "target_value": float(t), "policy_label": f"target_sensitivity@{t:.2f}"})
    for t in specificity_targets:
        jobs.append({"policy_group": "target_specificity", "target_value": float(t), "policy_label": f"target_specificity@{t:.2f}"})
    return jobs


def select_threshold_from_job(y_true, y_prob, job):
    policy_group = job["policy_group"]
    target_value = job["target_value"]
    policy_label = job["policy_label"]

    if policy_group == "youden":
        thr, sens, spec = youden_threshold(y_true, y_prob)
        fallback_used = False
        policy_label_applied = policy_label
    elif policy_group == "fixed":
        thr, sens, spec = threshold_fixed(y_true, y_prob, fixed_threshold=target_value)
        fallback_used = False
        policy_label_applied = policy_label
    elif policy_group == "target_sensitivity":
        thr, sens, spec, fallback_used = threshold_for_target_sensitivity(y_true, y_prob, target_sens=target_value)
        policy_label_applied = policy_label if not fallback_used else f"{policy_label}|fallback_youden"
    elif policy_group == "target_specificity":
        thr, sens, spec, fallback_used = threshold_for_target_specificity(y_true, y_prob, target_spec=target_value)
        policy_label_applied = policy_label if not fallback_used else f"{policy_label}|fallback_youden"
    else:
        raise ValueError(f"Unknown policy_group: {policy_group}")

    return {
        "policy_group": policy_group,
        "target_value": target_value,
        "policy_label_requested": policy_label,
        "policy_label_applied": policy_label_applied,
        "fallback_used": fallback_used,
        "threshold": thr,
        "sens_at_thr_train_oof": sens,
        "spec_at_thr_train_oof": spec,
    }
