from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

from utils.data import read_external_data, build_marker_frame
from utils.io_utils import load_bundle, save_excel, save_json, find_latest_result_dir
from utils.metrics import compute_metrics
from utils.thresholding import build_threshold_jobs
from utils.diagnostics import summarize_dual_zone, bootstrap_ci_metrics
from utils.plots import plot_roc, plot_pr, plot_calibration, plot_metrics_tradeoff, plot_diagnostic_zones_fixed
from utils.dca import build_dca_thresholds, decision_curve_df
from utils.markers import run_marker_analysis
from utils.shap_utils import compute_pipeline_shap, compute_catboost_shap


def _extract_probs(bundle: dict, X_ext: pd.DataFrame) -> np.ndarray:
    model = bundle["model"]
    return model.predict_proba(X_ext[bundle["feature_names"]])[:, 1]


def run_external_pipeline(cfg: dict) -> None:
    run_dir = Path(cfg["output"]["run_dir"])
    target_col = cfg["data"]["target_col"]

    bundle_cfg = cfg["internal_artifacts"]

    if "internal_results_root" in bundle_cfg:
        latest_internal_dir = find_latest_result_dir(bundle_cfg["internal_results_root"])
        print(f"[INFO] latest internal dir: {latest_internal_dir}")

        bundles = {
            "LogReg": load_bundle(latest_internal_dir / "logreg_bundle.joblib"),
            "RandomForest": load_bundle(latest_internal_dir / "randomforest_bundle.joblib"),
            "XGBoost": load_bundle(latest_internal_dir / "xgboost_bundle.joblib"),
            "CatBoost": load_bundle(latest_internal_dir / "catboost_bundle.joblib"),
        }

        internal_results_xlsx = latest_internal_dir / "internal_results.xlsx"
    else:
        bundles = {
            "LogReg": load_bundle(bundle_cfg["logreg_bundle_path"]),
            "RandomForest": load_bundle(bundle_cfg["randomforest_bundle_path"]),
            "XGBoost": load_bundle(bundle_cfg["xgboost_bundle_path"]),
            "CatBoost": load_bundle(bundle_cfg["catboost_bundle_path"]),
        }
        internal_results_xlsx = Path(bundle_cfg["internal_results_xlsx"])

    df_ext = read_external_data(cfg)
    y_ext = df_ext[target_col].astype(int).values

    prob_dict = {name: _extract_probs(bundle, df_ext) for name, bundle in bundles.items()}

    threshold_rows = []
    cm_rows = []
    pred_rows = []
    bootstrap_rows = []
    plot_rows = []
    diagnostic_rows = []
    diagnostic_detail_rows = []
    diagnostic_summary_rows = []

    fixed_thr = cfg["thresholds"]["fixed_threshold"]

    for name, bundle in bundles.items():
        y_prob = prob_dict[name]

        met, cm = compute_metrics(y_ext, y_prob, fixed_thr)
        threshold_rows.append({"model": name, "threshold": fixed_thr, **met})
        cm_rows.append({"model": name, "TN": int(cm[0,0]), "FP": int(cm[0,1]), "FN": int(cm[1,0]), "TP": int(cm[1,1])})

        pred_rows.extend([{"model": name, "y_true": int(y), "y_prob": float(p)} for y, p in zip(y_ext, y_prob)])

        ci = bootstrap_ci_metrics(y_ext, y_prob, fixed_thr, n_bootstrap=cfg["bootstrap"]["n_bootstrap"], seed=cfg["bootstrap"]["seed"])
        bootstrap_rows.append({"model": name, **{f"{k}_CI_low": v["ci_low"] for k, v in ci.items()}, **{f"{k}_CI_high": v["ci_high"] for k, v in ci.items()}})

        t_out = bundle["diagnostic_cutoffs"]["rule_out_cutoff"]
        t_in = bundle["diagnostic_cutoffs"]["rule_in_cutoff"]
        detail_df, summary, _ = summarize_dual_zone(y_ext, y_prob, t_out, t_in, dataset_name="external", object_name=name)

        roc_png = str((run_dir / f"{name}_EXTERNAL_ROC.png").resolve())
        pr_png = str((run_dir / f"{name}_EXTERNAL_PR.png").resolve())
        cal_png = str((run_dir / f"{name}_EXTERNAL_CAL.png").resolve())
        trade_png = str((run_dir / f"{name}_EXTERNAL_Tradeoff.png").resolve())
        zone_png = str((run_dir / f"{name}_EXTERNAL_Diagnostic_Zones.png").resolve())

        plot_roc(y_ext, y_prob, f"{name} EXTERNAL ROC", roc_png)
        plot_pr(y_ext, y_prob, f"{name} EXTERNAL PR", pr_png)
        plot_calibration(y_ext, y_prob, f"{name} EXTERNAL Calibration", cal_png, n_bins=cfg["plots"]["calibration_bins"])
        plot_metrics_tradeoff(y_ext, y_prob, f"{name} EXTERNAL Tradeoff", trade_png)
        plot_diagnostic_zones_fixed(y_ext, y_prob, name, zone_png, t_out, t_in, source_note="Internal test-derived cutoffs applied")

        diagnostic_rows.append({
            "Model": name,
            "Cutoff_Source": "internal_test_only",
            "Rule_out_Cutoff": t_out,
            "Rule_in_Cutoff": t_in,
            "Gray_n": summary["gray_n"],
            "Gray_pct": summary["gray_pct"],
            "Rule_out_NPV": summary["rule_out_npv"],
            "Rule_in_PPV": summary["rule_in_ppv"],
        })
        diagnostic_detail_rows.extend(detail_df.to_dict(orient="records"))
        diagnostic_summary_rows.append(summary)
        plot_rows.append({"model": name, "roc_png": roc_png, "pr_png": pr_png, "cal_png": cal_png, "tradeoff_png": trade_png, "zone_png": zone_png})

    external_metrics_df = pd.DataFrame(threshold_rows)
    external_cm_df = pd.DataFrame(cm_rows)
    external_pred_df = pd.DataFrame(pred_rows)
    external_bootstrap_df = pd.DataFrame(bootstrap_rows)
    external_diagnostic_df = pd.DataFrame(diagnostic_rows)
    external_dual_zone_detail_df = pd.DataFrame(diagnostic_detail_rows)
    external_dual_zone_summary_df = pd.DataFrame(diagnostic_summary_rows)
    plot_df = pd.DataFrame(plot_rows)

    # marker comparator
    marker_df_ext = build_marker_frame(df_ext)
    internal_marker_df = pd.read_excel(internal_results_xlsx, sheet_name=cfg["markers"]["internal_marker_sheet"])
    m2_row = internal_marker_df[internal_marker_df["marker"] == "M2BPGi"].iloc[0]

    marker_specs = [
        {"marker": "FIB-4", "rule_out_cutoff": cfg["markers"]["fib4_rule_out"], "rule_in_cutoff": cfg["markers"]["fib4_rule_in"], "source_note": "fixed cutoffs"},
        {"marker": "NFS", "rule_out_cutoff": cfg["markers"]["nfs_rule_out"], "rule_in_cutoff": cfg["markers"]["nfs_rule_in"], "source_note": "fixed cutoffs"},
        {"marker": "M2BPGi", "rule_out_cutoff": float(m2_row["rule_out_cutoff"]), "rule_in_cutoff": float(m2_row["rule_in_cutoff"]), "source_note": "internal test-derived cutoffs"},
    ]
    marker_results = run_marker_analysis(marker_df_ext, y_ext, marker_specs, run_dir, "marker", "external")

    # DCA
    dca_thresholds = build_dca_thresholds(cfg["plots"]["dca_threshold_min"], cfg["plots"]["dca_threshold_max"], cfg["plots"]["dca_threshold_step"])
    dca_frames = [decision_curve_df(y_ext, prob, dca_thresholds, model_name=name, dataset_name="external") for name, prob in prob_dict.items()]
    external_dca_df = pd.concat(dca_frames, ignore_index=True)

    # SHAP on external rows
    shap_lr = compute_pipeline_shap(bundles["LogReg"]["model"], df_ext[bundles["LogReg"]["feature_names"]], "LogReg")
    shap_rf = compute_pipeline_shap(bundles["RandomForest"]["model"], df_ext[bundles["RandomForest"]["feature_names"]], "RandomForest")
    shap_xgb = compute_pipeline_shap(bundles["XGBoost"]["model"], df_ext[bundles["XGBoost"]["feature_names"]], "XGBoost")
    shap_cat = compute_catboost_shap(bundles["CatBoost"]["model"], df_ext[bundles["CatBoost"]["feature_names"]])

    meta = {
        "mode": "external",
        "run_dir": str(run_dir),
        "diagnostic_zones": external_diagnostic_df.to_dict(orient="records"),
        "dual_zone_summary_external": external_dual_zone_summary_df.to_dict(orient="records"),
        "marker_diagnostic_external": marker_results["diagnostic_df"].to_dict(orient="records"),
        "marker_zone_summary_external": marker_results["zone_summary_df"].to_dict(orient="records"),
    }
    save_json(meta, run_dir / "external_meta.json")

    save_excel({
        "external_metrics": external_metrics_df,
        "external_cm_all": external_cm_df,
        "pred_external_all": external_pred_df,
        "external_bootstrap_all": external_bootstrap_df,
        "diagnostic_cutoffs_external": external_diagnostic_df,
        "external_dual_zone_detail": external_dual_zone_detail_df,
        "external_dual_zone_summary": external_dual_zone_summary_df,
        "plot_paths": plot_df,
        "marker_diag_external": marker_results["diagnostic_df"],
        "marker_zone_detail_ext": marker_results["zone_detail_df"],
        "marker_zone_summary_ext": marker_results["zone_summary_df"],
        "marker_plot_paths_ext": marker_results["plot_df"],
        "external_dca_long": external_dca_df,
        "shap_lr": shap_lr["df"],
        "shap_rf": shap_rf["df"],
        "shap_xgb": shap_xgb["df"],
        "shap_cat": shap_cat["df"],
    }, run_dir / "external_results.xlsx")

    print(f"[DONE] external pipeline finished: {run_dir}")
