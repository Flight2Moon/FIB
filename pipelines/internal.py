from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
import optuna

from utils.data import read_internal_data, select_features, build_marker_frame
from models.builders import build_models, get_default_param_spaces
from utils.metrics import compute_metrics
from utils.thresholding import build_threshold_jobs, select_threshold_from_job
from utils.diagnostics import find_rule_out_cutoff, find_rule_in_cutoff, summarize_dual_zone, bootstrap_ci_metrics
from utils.plots import plot_roc, plot_pr, plot_calibration, plot_metrics_tradeoff, plot_roc_with_ci, plot_diagnostic_zones_fixed
from utils.dca import build_dca_thresholds, decision_curve_df
from utils.shap_utils import compute_pipeline_shap, compute_catboost_shap
from utils.markers import find_marker_rule_out_cutoff, find_marker_rule_in_cutoff, run_marker_analysis
from utils.io_utils import save_json, save_excel, save_bundle


def _fit_with_optional(est, X, y, fit_params=None):
    if fit_params:
        est.fit(X, y, **fit_params)
    else:
        est.fit(X, y)
    return est


def _cross_validated_auc(estimator, X, y, cv, fit_params=None):
    from sklearn.metrics import roc_auc_score
    aucs = []
    oof_pred = np.full(len(y), np.nan, dtype=float)
    X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    y_s = pd.Series(y).reset_index(drop=True)

    for tr_idx, va_idx in cv.split(X_df, y_s):
        X_tr, X_va = X_df.iloc[tr_idx], X_df.iloc[va_idx]
        y_tr, y_va = y_s.iloc[tr_idx], y_s.iloc[va_idx]
        est = clone(estimator)
        est = _fit_with_optional(est, X_tr, y_tr, fit_params)
        if hasattr(est, "predict_proba"):
            va_prob = est.predict_proba(X_va)[:, 1]
        else:
            scores = est.decision_function(X_va)
            va_prob = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
        aucs.append(roc_auc_score(y_va, va_prob))
        oof_pred[va_idx] = va_prob
    return np.array(aucs), oof_pred


def _optuna_search(estimator, param_fn, X, y, cv, n_trials=50, fit_params=None,
                   random_state=42, use_stability_penalty=True, stability_penalty_weight=0.25):
    def objective(trial):
        resolved_params = param_fn(trial)

        current_est = clone(estimator)
        current_est.set_params(**resolved_params)

        cv_result = _cross_validated_auc(
            current_est,
            X,
            y,
            cv=cv,
            fit_params=fit_params
        )

        # `_cross_validated_auc` returns `(fold_aucs, oof_pred)`.
        # Be defensive in case it later returns just fold AUCs.
        if isinstance(cv_result, tuple):
            aucs = cv_result[0]
        else:
            aucs = cv_result

        aucs = np.asarray(aucs, dtype=float)
        mean_auc = float(np.mean(aucs))
        std_auc = float(np.std(aucs))

        trial.set_user_attr("mean_auc", mean_auc)
        trial.set_user_attr("std_auc", std_auc)
        trial.set_user_attr("resolved_params", resolved_params)

        return mean_auc - stability_penalty_weight * std_auc if use_stability_penalty else mean_auc

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_trial = study.best_trial
    best_params_resolved = best_trial.user_attrs["resolved_params"]

    best_est = clone(estimator)
    best_est.set_params(**best_params_resolved)

    if fit_params is None:
        best_est.fit(X, y)
    else:
        best_est.fit(X, y, **fit_params)

    return best_est, best_params_resolved, float(study.best_value), study

def run_internal_pipeline(cfg: dict) -> None:
    run_dir = Path(cfg["output"]["run_dir"])
    target_col = cfg["data"]["target_col"]
    df_train, df_test, df_var = read_internal_data(cfg)
    selected = select_features(df_train, df_test, df_var, cfg)
    X_train, X_test = selected["X_train"], selected["X_test"]
    y_train, y_test = selected["y_train"], selected["y_test"]
    num_cols, cat_cols, all_features = selected["num_cols"], selected["cat_cols"], selected["all_features"]
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)

    models = build_models(num_cols, cat_cols, scale_pos_weight, cfg["cv"]["random_state"], cfg["cv"]["n_jobs"], disable_catboost_files=True)
    param_spaces = get_default_param_spaces(scale_pos_weight)
    cv = StratifiedKFold(n_splits=cfg["cv"]["n_splits"], shuffle=True, random_state=cfg["cv"]["random_state"])

    cat_idx = [X_train.columns.get_loc(c) for c in cat_cols]

    fit_param_map = {
        "LogReg": None,
        "RandomForest": None,
        "XGBoost": None,
        "CatBoost": {"cat_features": cat_idx},
    }
    search_trials = {
        "LogReg": cfg["search"]["search_n_iter_lr"],
        "RandomForest": cfg["search"]["search_n_iter_rf"],
        "XGBoost": cfg["search"]["search_n_iter_xgb"],
        "CatBoost": cfg["search"]["search_n_iter_cat"],
    }

    fitted = {}
    best_params = {}
    best_objective = {}
    studies = {}
    oof_dict = {}

    for name, est in models.items():
        fitted[name], best_params[name], best_objective[name], studies[name] = _optuna_search(
            est, param_spaces[name], X_train, y_train, cv,
            n_trials=search_trials[name], fit_params=fit_param_map[name],
            random_state=cfg["cv"]["random_state"],
        )
        _, oof_pred = _cross_validated_auc(fitted[name], X_train, y_train, cv, fit_params=fit_param_map[name])
        oof_dict[name] = oof_pred

    threshold_jobs = build_threshold_jobs(cfg["thresholds"]["sensitivity_targets"], cfg["thresholds"]["specificity_targets"], cfg["thresholds"]["fixed_threshold"])
    threshold_rows, test_point_rows, bootstrap_rows = [], [], []

    threshold_export_cols = ["policy_group", "target_value", "policy_label_applied", "threshold", "sens_at_thr_train_oof", "spec_at_thr_train_oof", "fallback_used"]
    representative_policy = cfg["thresholds"]["representative_threshold_policy"]
    best_threshold_by_model = {}

    test_prob_dict = {}
    for name, est in fitted.items():
        if name == "CatBoost":
            test_prob = est.predict_proba(X_test)[:, 1]
        else:
            test_prob = est.predict_proba(X_test)[:, 1]
        test_prob_dict[name] = test_prob

        for idx_job, job in enumerate(threshold_jobs):
            sel = select_threshold_from_job(y_train, oof_dict[name], job)
            threshold_rows.append({"model": name, **sel})

            met, _ = compute_metrics(y_test, test_prob, sel["threshold"])
            test_point_rows.append({"model": name, **sel, **met})

            ci = bootstrap_ci_metrics(y_test, test_prob, sel["threshold"], n_bootstrap=cfg["bootstrap"]["n_bootstrap"], seed=cfg["bootstrap"]["seed"] + idx_job)
            row_ci = {"model": name, **sel}
            for k, v in ci.items():
                row_ci[f"{k}_CI_low"] = v["ci_low"]
                row_ci[f"{k}_CI_high"] = v["ci_high"]
                row_ci[f"{k}_n_valid"] = v["n_valid"]
            bootstrap_rows.append(row_ci)

        rep = [r for r in threshold_rows if r["model"] == name and r["policy_group"] == representative_policy]
        best_threshold_by_model[name] = rep[0]["threshold"] if rep else 0.5

    thresholds_df = pd.DataFrame(threshold_rows)
    test_point_df = pd.DataFrame(test_point_rows)
    test_bootstrap_df = pd.DataFrame(bootstrap_rows)

    diagnostic_results, diagnostic_zone_detail_rows, diagnostic_zone_summary_rows, plot_rows = [], [], [], []
    for name, y_prob in test_prob_dict.items():
        t_out, out_row, out_fb = find_rule_out_cutoff(y_test, y_prob, target_sens=cfg["thresholds"]["diagnostic_target_sens"])
        t_in, in_row, in_fb = find_rule_in_cutoff(y_test, y_prob, target_spec=cfg["thresholds"]["diagnostic_target_spec"])
        zone_df, zone_summary, _ = summarize_dual_zone(y_test, y_prob, t_out, t_in, dataset_name="internal_test", object_name=name)

        roc_png = str((run_dir / f"{name}_TEST_ROC.png").resolve())
        pr_png = str((run_dir / f"{name}_TEST_PR.png").resolve())
        cal_png = str((run_dir / f"{name}_TEST_Calibration.png").resolve())
        tradeoff_png = str((run_dir / f"{name}_TEST_Tradeoff.png").resolve())
        zone_png = str((run_dir / f"{name}_TEST_Diagnostic_Zones.png").resolve())

        plot_roc(y_test, y_prob, f"{name} TEST ROC", roc_png)
        plot_pr(y_test, y_prob, f"{name} TEST PR", pr_png)
        plot_calibration(y_test, y_prob, f"{name} TEST Calibration", cal_png, n_bins=cfg["plots"]["calibration_bins"])
        plot_metrics_tradeoff(y_test, y_prob, f"{name} Metrics vs Threshold", tradeoff_png)
        plot_diagnostic_zones_fixed(y_test, y_prob, object_name=name, out_png=zone_png, t_out=t_out, t_in=t_in, source_note="Internal test-derived cutoffs")

        diagnostic_results.append({
            "Model": name,
            "Cutoff_Source": "test_only_internal",
            "Rule_out_Cutoff": t_out,
            "Rule_in_Cutoff": t_in,
            "Target_Sensitivity": cfg["thresholds"]["diagnostic_target_sens"],
            "Target_Specificity": cfg["thresholds"]["diagnostic_target_spec"],
            "Rule_out_Fallback_Used": out_fb,
            "Rule_in_Fallback_Used": in_fb,
            "Rule_out_Sens_At_Cutoff": out_row["sensitivity"],
            "Rule_out_Spec_At_Cutoff": out_row["specificity"],
            "Rule_in_Sens_At_Cutoff": in_row["sensitivity"],
            "Rule_in_Spec_At_Cutoff": in_row["specificity"],
            "Rule_out_NPV": zone_summary["rule_out_npv"],
            "Rule_out_Sensitivity": zone_summary["rule_out_sensitivity"],
            "Rule_in_PPV": zone_summary["rule_in_ppv"],
            "Rule_in_Specificity": zone_summary["rule_in_specificity"],
            "Gray_n": zone_summary["gray_n"],
            "Gray_pct": zone_summary["gray_pct"],
            "Zone_Plot_Path": zone_png,
        })
        diagnostic_zone_detail_rows.extend(zone_df.to_dict(orient="records"))
        diagnostic_zone_summary_rows.append(zone_summary)
        plot_rows.append({"model": name, "roc_png": roc_png, "pr_png": pr_png, "cal_png": cal_png, "tradeoff_png": tradeoff_png, "zone_png": zone_png})

    diagnostic_df = pd.DataFrame(diagnostic_results)
    diagnostic_zone_detail_df = pd.DataFrame(diagnostic_zone_detail_rows)
    diagnostic_zone_summary_df = pd.DataFrame(diagnostic_zone_summary_rows)
    plot_df = pd.DataFrame(plot_rows)

    # Marker analysis
    marker_df_test = build_marker_frame(df_test)
    m2_out, _, _ = find_marker_rule_out_cutoff(y_test, pd.to_numeric(marker_df_test["M2BPGi"], errors="coerce").values, target_sens=cfg["thresholds"]["diagnostic_target_sens"])
    m2_in, _, _ = find_marker_rule_in_cutoff(y_test, pd.to_numeric(marker_df_test["M2BPGi"], errors="coerce").values, target_spec=cfg["thresholds"]["diagnostic_target_spec"])
    marker_specs = [
        {"marker": "FIB-4", "rule_out_cutoff": cfg["markers"]["fib4_rule_out"], "rule_in_cutoff": cfg["markers"]["fib4_rule_in"], "source_note": "fixed cutoffs"},
        {"marker": "NFS", "rule_out_cutoff": cfg["markers"]["nfs_rule_out"], "rule_in_cutoff": cfg["markers"]["nfs_rule_in"], "source_note": "fixed cutoffs"},
        {"marker": "M2BPGi", "rule_out_cutoff": m2_out, "rule_in_cutoff": m2_in, "source_note": "internal sens95/spec90"},
    ]
    marker_results = run_marker_analysis(marker_df_test, y_test, marker_specs, run_dir, "marker", "internal_test")

    # DCA
    dca_thresholds = build_dca_thresholds(cfg["plots"]["dca_threshold_min"], cfg["plots"]["dca_threshold_max"], cfg["plots"]["dca_threshold_step"])
    dca_frames = [decision_curve_df(y_test, prob, dca_thresholds, model_name=name, dataset_name="internal_test") for name, prob in test_prob_dict.items()]
    dca_df = pd.concat(dca_frames, ignore_index=True)

    # SHAP
    shap_cfg = cfg["shap"]
    shap_lr = compute_pipeline_shap(fitted["LogReg"], X_train, "LogReg", shap_cfg["max_samples"], shap_cfg["lr_bg_n"], shap_cfg["group_to_original_feature"])
    shap_rf = compute_pipeline_shap(fitted["RandomForest"], X_train, "RandomForest", shap_cfg["max_samples"], shap_cfg["lr_bg_n"], shap_cfg["group_to_original_feature"])
    shap_xgb = compute_pipeline_shap(fitted["XGBoost"], X_train, "XGBoost", shap_cfg["max_samples"], shap_cfg["lr_bg_n"], shap_cfg["group_to_original_feature"])
    shap_cat = compute_catboost_shap(fitted["CatBoost"], X_train, shap_cfg["max_samples"])

    diag_cutoff_map = {
        row["Model"]: {
            "source_dataset": row["Cutoff_Source"],
            "target_sensitivity": row["Target_Sensitivity"],
            "target_specificity": row["Target_Specificity"],
            "rule_out_cutoff": row["Rule_out_Cutoff"],
            "rule_in_cutoff": row["Rule_in_Cutoff"],
        } for _, row in diagnostic_df.iterrows()
    }

    bundle_paths = {}
    for name in fitted:
        bundle = {
            "model": fitted[name],
            "model_name": name,
            "feature_names": all_features,
            "num_cols": num_cols,
            "cat_cols": cat_cols,
            "representative_threshold_policy": representative_policy,
            "best_threshold": best_threshold_by_model[name],
            "all_thresholds": thresholds_df.loc[thresholds_df["model"] == name, threshold_export_cols].to_dict(orient="records"),
            "diagnostic_cutoffs": diag_cutoff_map[name],
        }
        if name == "CatBoost":
            bundle["cat_feature_indices"] = [X_train.columns.get_loc(c) for c in cat_cols]
        path = run_dir / f"{name.lower()}_bundle.joblib"
        save_bundle(bundle, path)
        bundle_paths[name] = str(path)

    meta = {
        "mode": "internal",
        "run_dir": str(run_dir),
        "all_features": all_features,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "optuna_best_params": best_params,
        "optuna_best_objective_value": best_objective,
        "diagnostic_cutoffs_by_model": diag_cutoff_map,
        "marker_diagnostic_internal": marker_results["diagnostic_df"].to_dict(orient="records"),
        "marker_zone_summary_internal": marker_results["zone_summary_df"].to_dict(orient="records"),
        "bundle_paths": bundle_paths,
    }
    save_json(meta, run_dir / "internal_meta.json")

    save_excel({
        "thresholds_all": thresholds_df,
        "test_point_all": test_point_df,
        "test_bootstrap_all": test_bootstrap_df,
        "diagnostic_cutoffs_internal": diagnostic_df,
        "diagnostic_zone_detail": diagnostic_zone_detail_df,
        "diagnostic_zone_summary": diagnostic_zone_summary_df,
        "plot_paths": plot_df,
        "marker_diag_internal": marker_results["diagnostic_df"],
        "marker_zone_detail_int": marker_results["zone_detail_df"],
        "marker_zone_summary_int": marker_results["zone_summary_df"],
        "marker_plot_paths_int": marker_results["plot_df"],
        "dca_internal_long": dca_df,
        "shap_lr": shap_lr["df"],
        "shap_rf": shap_rf["df"],
        "shap_xgb": shap_xgb["df"],
        "shap_cat": shap_cat["df"],
    }, run_dir / "internal_results.xlsx")

    print(f"[DONE] internal pipeline finished: {run_dir}")
