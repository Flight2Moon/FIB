from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from scipy import sparse


def _safe_sample_df(X, n, random_state=42):
    if len(X) <= n:
        return X.copy()
    return X.sample(n=n, random_state=random_state).copy()


def _make_shap_summary_df(feature_names, shap_values):
    mean_abs = np.abs(shap_values).mean(axis=0)
    out = pd.DataFrame({"feature": list(feature_names), "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    total = out["mean_abs_shap"].sum()
    out["normalized_sum1"] = out["mean_abs_shap"] / total if total > 0 else 0.0
    return out


def _group_ohe_shap_to_original_features(feature_names, shap_values):
    if shap_values.ndim != 2:
        raise ValueError(f"_group_ohe_shap_to_original_features expects 2D shap_values, got shape={shap_values.shape}")
    grouped = {}
    for j, fname in enumerate(feature_names):
        fname = str(fname)
        if fname.startswith("num__"):
            base = fname.replace("num__", "")
        elif fname.startswith("cat__"):
            base = fname.replace("cat__", "")
            if "_" in base:
                base = "_".join(base.split("_")[:-1]) or base
        else:
            base = fname
        if base not in grouped:
            grouped[base] = np.zeros(shap_values.shape[0], dtype=float)
        grouped[base] += shap_values[:, j]
    grouped_names = list(grouped.keys())
    grouped_matrix = np.column_stack([grouped[k] for k in grouped_names])
    return grouped_names, grouped_matrix


def compute_pipeline_shap(fitted_pipeline, X_train, model_name, shap_max_samples=800, shap_lr_bg_n=200, group_to_original_feature=True, random_state=42):
    X_use = _safe_sample_df(X_train, shap_max_samples, random_state=random_state)
    preprocess = fitted_pipeline.named_steps["preprocess"]
    model = fitted_pipeline.named_steps["model"]
    X_trans = preprocess.transform(X_use)
    feature_names = preprocess.get_feature_names_out()

    if model_name == "LogReg":
        bg_df = _safe_sample_df(X_train, shap_lr_bg_n, random_state=random_state)
        bg_trans = preprocess.transform(bg_df)

        if sparse.issparse(bg_trans):
            bg_trans = bg_trans.tocsr()
        if sparse.issparse(X_trans):
            X_eval = X_trans.tocsr()
        else:
            X_eval = np.asarray(X_trans)

        explainer = shap.LinearExplainer(model, bg_trans)
        shap_values = explainer.shap_values(X_eval)

    else:
        X_dense = X_trans.toarray() if sparse.issparse(X_trans) else np.asarray(X_trans)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_dense)

    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    shap_values = np.asarray(shap_values)

    # SHAP for some tree classifiers can come back as
    # (n_samples, n_features, n_outputs). Reduce to positive class.
    if shap_values.ndim == 3:
        if shap_values.shape[2] == 2:
            shap_values = shap_values[:, :, 1]
        else:
            shap_values = shap_values[:, :, -1]
    elif shap_values.ndim != 2:
        raise ValueError(f"Unexpected SHAP shape: {shap_values.shape}")

    if group_to_original_feature:
        grouped_names, grouped_shap = _group_ohe_shap_to_original_features(feature_names, shap_values)
        summary_df = _make_shap_summary_df(grouped_names, grouped_shap)
        return {"ok": True, "df": summary_df, "feature_names": grouped_names, "shap_values": grouped_shap}

    summary_df = _make_shap_summary_df(feature_names, shap_values)
    return {"ok": True, "df": summary_df, "feature_names": list(feature_names), "shap_values": shap_values}


def compute_catboost_shap(fitted_cat, X_train, shap_max_samples=800, random_state=42):
    X_use = _safe_sample_df(X_train, shap_max_samples, random_state=random_state)
    explainer = shap.TreeExplainer(fitted_cat)
    shap_values = explainer.shap_values(X_use)
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    shap_values = np.asarray(shap_values)
    summary_df = _make_shap_summary_df(list(X_use.columns), shap_values)
    return {"ok": True, "df": summary_df, "feature_names": list(X_use.columns), "shap_values": shap_values}
