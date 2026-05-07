from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from utils.metrics import compute_specificity
from utils.diagnostics import build_sens_spec_table

SENS_COLOR = "#00AEEF"
SPEC_COLOR = "#F15A24"


def plot_roc(y_true, y_prob, title, out_png):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color="black", lw=2, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("1-Specificity")
    plt.ylabel("Sensitivity")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_pr(y_true, y_prob, title, out_png):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, color="black", lw=2, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_calibration(y_true, y_prob, title, out_png, n_bins=10):
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    brier = brier_score_loss(y_true, y_prob)
    plt.figure(figsize=(6, 6))
    plt.plot(mean_pred, frac_pos, marker="o", lw=2, label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(f"{title}\nBrier = {brier:.3f}")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_roc_with_ci(y_true, y_prob, title, out_png, ci_low, ci_high):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='black', lw=2)
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    text_str = f"Area under ROC curve = {auc:.3f}\n95% CI : {ci_low:.3f}-{ci_high:.3f}"
    plt.gca().text(0.55, 0.1, text_str, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
    plt.xlabel("1-Specificity")
    plt.ylabel("Sensitivity")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_metrics_tradeoff(y_true, y_prob, title, out_png):
    from sklearn.metrics import confusion_matrix, recall_score
    thresholds = np.linspace(0, 1, 100)
    sens, spec, ppv, npv = [], [], [], []
    for thr in thresholds:
        y_pred = (np.asarray(y_prob) >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sens.append(tp / (tp + fn) if (tp + fn) > 0 else np.nan)
        spec.append(tn / (tn + fp) if (tn + fp) > 0 else np.nan)
        ppv.append(tp / (tp + fp) if (tp + fp) > 0 else np.nan)
        npv.append(tn / (tn + fn) if (tn + fn) > 0 else np.nan)
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, sens, label="Sensitivity", lw=2)
    plt.plot(thresholds, spec, label="Specificity", lw=2)
    plt.plot(thresholds, ppv, label="PPV", lw=2)
    plt.plot(thresholds, npv, label="NPV", lw=2)
    plt.xlabel("Threshold")
    plt.ylabel("Metric value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_diagnostic_zones_fixed(y_true, y_prob, object_name, out_png, t_out, t_in, source_note="", dataset_label=None, show_gray_metrics=True):
    thresholds = np.linspace(0, 1, 1001)
    sens_list, spec_list = [], []
    for thr in thresholds:
        y_pred = (np.asarray(y_prob) >= thr).astype(int)
        from sklearn.metrics import confusion_matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sens_list.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        spec_list.append(tn / (tn + fp) if (tn + fp) > 0 else 0)

    ss = build_sens_spec_table(y_true, y_prob, thresholds=[t_out, t_in])
    out_metrics = ss.iloc[0].to_dict()
    in_metrics = ss.iloc[1].to_dict()

    y_prob_arr = np.asarray(y_prob)
    gray_mask = (y_prob_arr >= t_out) & (y_prob_arr < t_in)
    gray_n = int(np.sum(gray_mask))
    gray_pct = float(gray_n / len(y_prob_arr)) if len(y_prob_arr) > 0 else np.nan

    if object_name in {"FIB-4", "NFS", "M2BPGi"}:
        x_label = f"{object_name} threshold"
    else:
        x_label = "Predicted probability"

    if dataset_label:
        plot_title = f"[{object_name}] {dataset_label} Diagnostic Zones"
    else:
        plot_title = f"[{object_name}] Diagnostic Zones"

    plt.figure(figsize=(12, 6))
    plt.plot(thresholds, sens_list, label="Sensitivity", linestyle="--", color=SENS_COLOR, lw=2)
    plt.plot(thresholds, spec_list, label="Specificity", linestyle="--", color=SPEC_COLOR, lw=2)
    plt.axvspan(0, t_out, color='green', alpha=0.1, label='Rule-out Zone')
    plt.axvspan(t_out, t_in, color='gray', alpha=0.1, label='Gray Zone')
    plt.axvspan(t_in, 1, color='red', alpha=0.1, label='Rule-in Zone')
    plt.axvline(t_out, color='green', linestyle=':', lw=1.5)
    plt.axvline(t_in, color='red', linestyle=':', lw=1.5)
    plt.scatter([t_out], [out_metrics["sensitivity"]], color=SENS_COLOR, s=55, zorder=5)
    plt.scatter([t_out], [out_metrics["specificity"]], color=SPEC_COLOR, s=55, zorder=5)
    plt.scatter([t_in], [in_metrics["sensitivity"]], color=SENS_COLOR, s=55, zorder=5)
    plt.scatter([t_in], [in_metrics["specificity"]], color=SPEC_COLOR, s=55, zorder=5)
    plt.annotate(f"R/O={t_out:.4f}\nSens={out_metrics['sensitivity']:.3f}\nSpec={out_metrics['specificity']:.3f}",
                 xy=(t_out, out_metrics["sensitivity"]), xytext=(t_out + 0.03, min(0.98, out_metrics["sensitivity"] + 0.08)),
                 fontsize=9, color="green", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="green", alpha=0.8))
    plt.annotate(f"R/I={t_in:.4f}\nSens={in_metrics['sensitivity']:.3f}\nSpec={in_metrics['specificity']:.3f}",
                 xy=(t_in, in_metrics["specificity"]), xytext=(max(0.02, t_in - 0.20), max(0.05, in_metrics["specificity"] - 0.18)),
                 fontsize=9, color="red", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="red", alpha=0.8))
    if source_note:
        plt.title(f"{plot_title}\n{source_note}")
    else:
        plt.title(plot_title)
    plt.xlabel(x_label)
    plt.ylabel("Performance Value")
    plt.ylim(0, 1.02)
    if show_gray_metrics:
        gray_text = f"Gray zone\nn = {gray_n}\n% = {gray_pct:.3f}"
        plt.gca().text(
            0.02,
            0.98,
            gray_text,
            transform=plt.gca().transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.85),
        )
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=5)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
