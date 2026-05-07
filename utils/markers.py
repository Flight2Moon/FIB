from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from utils.plots import plot_diagnostic_zones_fixed


def compute_marker_sens_spec_at_cutoff(y_true, marker_values, cutoff, positive_if="high"):
    y_true = np.asarray(y_true)
    marker_values = np.asarray(marker_values)
    if positive_if == "high":
        y_pred = (marker_values >= cutoff).astype(int)
    elif positive_if == "low":
        y_pred = (marker_values <= cutoff).astype(int)
    else:
        raise ValueError("positive_if must be 'high' or 'low'")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    return {"cutoff": float(cutoff), "sensitivity": sens, "specificity": spec, "ppv": ppv, "npv": npv, "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}


def find_marker_rule_out_cutoff(y_true, marker_values, target_sens=0.95, positive_if="high"):
    vals = np.sort(np.unique(marker_values[~pd.isnull(marker_values)]))
    rows = [compute_marker_sens_spec_at_cutoff(y_true, marker_values, c, positive_if=positive_if) for c in vals]
    ss_df = pd.DataFrame(rows)
    valid = ss_df[ss_df["sensitivity"] >= target_sens].copy()
    if len(valid) == 0:
        row = ss_df.loc[ss_df["sensitivity"].idxmax()]
        return float(row["cutoff"]), row.to_dict(), True
    row = valid.sort_values(["cutoff", "specificity"], ascending=[True, False]).iloc[-1]
    return float(row["cutoff"]), row.to_dict(), False


def find_marker_rule_in_cutoff(y_true, marker_values, target_spec=0.90, positive_if="high"):
    vals = np.sort(np.unique(marker_values[~pd.isnull(marker_values)]))
    rows = [compute_marker_sens_spec_at_cutoff(y_true, marker_values, c, positive_if=positive_if) for c in vals]
    ss_df = pd.DataFrame(rows)
    valid = ss_df[ss_df["specificity"] >= target_spec].copy()
    if len(valid) == 0:
        row = ss_df.loc[ss_df["specificity"].idxmax()]
        return float(row["cutoff"]), row.to_dict(), True
    row = valid.sort_values(["cutoff", "sensitivity"], ascending=[True, False]).iloc[0]
    return float(row["cutoff"]), row.to_dict(), False


def assign_marker_dual_zone(marker_values, t_out, t_in):
    marker_values = np.asarray(marker_values)
    zone_idx = np.full(len(marker_values), 1, dtype=int)
    zone_idx[marker_values < t_out] = 0
    zone_idx[marker_values > t_in] = 2
    return zone_idx


def summarize_marker_dual_zone(y_true, marker_values, t_out, t_in, marker_name, dataset_name):
    y_true = np.asarray(y_true)
    marker_values = np.asarray(marker_values)
    zone_idx = assign_marker_dual_zone(marker_values, t_out, t_in)
    detail_rows = []
    for z, zname in {0: "rule_out", 1: "gray", 2: "rule_in"}.items():
        mask = zone_idx == z
        n = int(mask.sum())
        pos = int((y_true[mask] == 1).sum()) if n > 0 else 0
        neg = int((y_true[mask] == 0).sum()) if n > 0 else 0
        detail_rows.append({"marker": marker_name, "dataset": dataset_name, "zone": zname, "n": n, "n_pct": n / len(y_true) if len(y_true) > 0 else np.nan, "pos": pos, "neg": neg, "pos_rate": pos / n if n > 0 else np.nan})

    out_metrics = compute_marker_sens_spec_at_cutoff(y_true, marker_values, t_out, positive_if="high")
    in_metrics = compute_marker_sens_spec_at_cutoff(y_true, marker_values, t_in, positive_if="high")
    summary = {
        "marker": marker_name,
        "dataset": dataset_name,
        "rule_out_cutoff": float(t_out),
        "rule_in_cutoff": float(t_in),
        "rule_out_sensitivity": out_metrics["sensitivity"],
        "rule_out_specificity": out_metrics["specificity"],
        "rule_out_npv": out_metrics["npv"],
        "rule_out_fn": out_metrics["FN"],
        "rule_in_sensitivity": in_metrics["sensitivity"],
        "rule_in_specificity": in_metrics["specificity"],
        "rule_in_ppv": in_metrics["ppv"],
        "rule_in_fp": in_metrics["FP"],
        "gray_n": int((zone_idx == 1).sum()),
        "gray_pct": float((zone_idx == 1).mean()),
    }
    return pd.DataFrame(detail_rows), summary, zone_idx


def run_marker_analysis(df_marker, y_true, specs, run_dir, prefix, dataset_name):
    diag_rows, zone_detail_rows, zone_summary_rows, plot_rows = [], [], [], []
    for spec in specs:
        marker_name = spec["marker"]
        vals = pd.to_numeric(df_marker[marker_name], errors="coerce").values
        mask = ~pd.isnull(vals) & ~pd.isnull(y_true)
        y_use = np.asarray(y_true)[mask]
        v_use = vals[mask]
        t_out = float(spec["rule_out_cutoff"])
        t_in = float(spec["rule_in_cutoff"])
        plot_path = str((run_dir / f"{prefix}_{marker_name}_{dataset_name}_Diagnostic_Zones.png").resolve())

        detail_df, summary, _ = summarize_marker_dual_zone(
            y_true=y_use,
            marker_values=v_use,
            t_out=t_out,
            t_in=t_in,
            marker_name=marker_name,
            dataset_name=dataset_name,
        )

        plot_diagnostic_zones_fixed(
            y_true=y_use,
            y_prob=v_use,
            object_name=marker_name,
            out_png=plot_path,
            t_out=t_out,
            t_in=t_in,
            source_note=spec.get("source_note", "")
        )

        auc = roc_auc_score(y_use, v_use) if len(np.unique(y_use)) >= 2 else np.nan
        ap = average_precision_score(y_use, v_use) if len(np.unique(y_use)) >= 2 else np.nan

        zone_detail_rows.extend(detail_df.to_dict(orient="records"))
        zone_summary_rows.append(summary)
        diag_rows.append({
            "marker": marker_name,
            "dataset": dataset_name,
            "rule_out_cutoff": t_out,
            "rule_in_cutoff": t_in,
            "source_note": spec.get("source_note", ""),
            "AUROC": auc,
            "PR_AUC(AP)": ap,
            "gray_n": summary["gray_n"],
            "gray_pct": summary["gray_pct"],
            "plot_path": plot_path,
        })
        plot_rows.append({"marker": marker_name, "dataset": dataset_name, "plot_path": plot_path})

    return {
        "diagnostic_df": pd.DataFrame(diag_rows),
        "zone_detail_df": pd.DataFrame(zone_detail_rows),
        "zone_summary_df": pd.DataFrame(zone_summary_rows),
        "plot_df": pd.DataFrame(plot_rows),
    }
