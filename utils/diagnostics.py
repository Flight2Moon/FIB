from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from utils.metrics import compute_metrics


def build_sens_spec_table(y_true, y_prob, thresholds=None):
    if thresholds is None:
        thresholds = np.unique(np.round(y_prob, 6))
        thresholds = np.concatenate(([0.0], thresholds, [1.0]))
        thresholds = np.unique(np.clip(thresholds, 0, 1))
    rows = []
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        rows.append({"threshold": float(thr), "sensitivity": sens, "specificity": spec, "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)})
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def find_rule_out_cutoff(y_true, y_prob, target_sens=0.95):
    ss_df = build_sens_spec_table(y_true, y_prob)
    valid = ss_df[ss_df["sensitivity"] >= target_sens].copy()
    if len(valid) == 0:
        best_idx = ss_df["sensitivity"].idxmax()
        row = ss_df.loc[best_idx]
        return float(row["threshold"]), row.to_dict(), True
    row = valid.sort_values(["threshold", "specificity"], ascending=[True, False]).iloc[-1]
    return float(row["threshold"]), row.to_dict(), False


def find_rule_in_cutoff(y_true, y_prob, target_spec=0.90):
    ss_df = build_sens_spec_table(y_true, y_prob)
    valid = ss_df[ss_df["specificity"] >= target_spec].copy()
    if len(valid) == 0:
        best_idx = ss_df["specificity"].idxmax()
        row = ss_df.loc[best_idx]
        return float(row["threshold"]), row.to_dict(), True
    row = valid.sort_values(["threshold", "sensitivity"], ascending=[True, False]).iloc[0]
    return float(row["threshold"]), row.to_dict(), False


def assign_dual_zone(y_prob, t_out, t_in):
    zones = np.full(len(y_prob), 1, dtype=int)
    zones[np.asarray(y_prob) < t_out] = 0
    zones[np.asarray(y_prob) > t_in] = 2
    return zones


def summarize_dual_zone(y_true, y_prob, t_out, t_in, dataset_name="internal_test", object_name="model"):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    zone_idx = assign_dual_zone(y_prob, t_out, t_in)
    zone_map = {0: "rule_out", 1: "gray", 2: "rule_in"}
    records = []
    for z, zname in zone_map.items():
        mask = zone_idx == z
        n = int(mask.sum())
        pos = int((y_true[mask] == 1).sum()) if n > 0 else 0
        neg = int((y_true[mask] == 0).sum()) if n > 0 else 0
        records.append({
            "object_name": object_name,
            "dataset": dataset_name,
            "zone": zname,
            "n": n,
            "n_pct": n / len(y_true) if len(y_true) > 0 else np.nan,
            "pos": pos,
            "neg": neg,
            "pos_rate": pos / n if n > 0 else np.nan,
        })

    mask_out = y_prob < t_out
    tn_out = int(((y_true == 0) & mask_out).sum()) if mask_out.sum() > 0 else 0
    fn_out = int(((y_true == 1) & mask_out).sum()) if mask_out.sum() > 0 else 0
    npv_out = tn_out / (tn_out + fn_out) if (tn_out + fn_out) > 0 else np.nan
    sens_out = 1 - (fn_out / int((y_true == 1).sum())) if int((y_true == 1).sum()) > 0 else np.nan

    mask_in = y_prob > t_in
    tp_in = int(((y_true == 1) & mask_in).sum()) if mask_in.sum() > 0 else 0
    fp_in = int(((y_true == 0) & mask_in).sum()) if mask_in.sum() > 0 else 0
    ppv_in = tp_in / (tp_in + fp_in) if (tp_in + fp_in) > 0 else np.nan
    spec_in = 1 - (fp_in / int((y_true == 0).sum())) if int((y_true == 0).sum()) > 0 else np.nan

    summary = {
        "object_name": object_name,
        "dataset": dataset_name,
        "rule_out_cutoff": float(t_out),
        "rule_in_cutoff": float(t_in),
        "rule_out_npv": npv_out,
        "rule_out_sensitivity": sens_out,
        "rule_out_fn": fn_out,
        "rule_in_ppv": ppv_in,
        "rule_in_specificity": spec_in,
        "rule_in_fp": fp_in,
        "gray_n": int((zone_idx == 1).sum()),
        "gray_pct": float((zone_idx == 1).mean()),
    }
    return pd.DataFrame(records), summary, zone_idx


def bootstrap_ci_metrics(y_true, y_prob, threshold, n_bootstrap=1000, seed=2026):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    buckets = {"AUC": [], "PR_AUC(AP)": [], "F1": [], "Recall(Sensitivity)": [], "Specificity": [], "Accuracy": []}
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        y_b = y_true[idx]
        p_b = y_prob[idx]
        if len(np.unique(y_b)) < 2:
            continue
        met, _ = compute_metrics(y_b, p_b, threshold)
        for k in buckets:
            buckets[k].append(met[k])

    def ci95(vals):
        vals = np.asarray(vals, dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            return {"ci_low": np.nan, "ci_high": np.nan, "n_valid": 0}
        return {
            "ci_low": float(np.percentile(vals, 2.5)),
            "ci_high": float(np.percentile(vals, 97.5)),
            "n_valid": int(len(vals)),
        }

    return {k: ci95(v) for k, v in buckets.items()}
