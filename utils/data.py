from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def read_internal_data(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_cfg = cfg["data"]
    df_train = pd.read_excel(data_cfg["excel_path"], sheet_name=data_cfg["train_sheet"])
    df_test = pd.read_excel(data_cfg["excel_path"], sheet_name=data_cfg["test_sheet"])
    df_var = pd.read_excel(data_cfg["excel_path"], sheet_name=data_cfg["var_sheet"])
    df_var["variable"] = df_var["variable"].astype(str)
    return df_train, df_test, df_var


def select_features(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    df_var: pd.DataFrame,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    fcfg = cfg["features"]
    target_col = cfg["data"]["target_col"]
    base_exclude_cols = list(fcfg.get("base_exclude_cols", []))

    num_cols_all = df_var.loc[df_var["class"] == 0, "variable"].tolist()
    cat_cols_all = df_var.loc[df_var["class"] == 1, "variable"].tolist()

    mode = fcfg["feature_mode"]
    if mode == "all":
        include_features = None
    elif mode == "5var":
        include_features = fcfg["features_5var"]
    elif mode == "6var":
        include_features = fcfg["features_6var"]
    elif mode == "custom":
        include_features = fcfg["custom_include"]
    else:
        raise ValueError(f"Unknown feature_mode: {mode}")

    candidate_features = [
        c
        for c in df_var["variable"].tolist()
        if c not in base_exclude_cols + [target_col]
        and c in df_train.columns
        and c in df_test.columns
    ]

    all_features = candidate_features if include_features is None else [c for c in include_features if c in candidate_features]
    if not all_features:
        raise ValueError("No usable features selected.")

    num_cols = [c for c in num_cols_all if c in all_features]
    cat_cols = [c for c in cat_cols_all if c in all_features]

    X_train = df_train[all_features].copy()
    X_test = df_test[all_features].copy()
    y_train = df_train[target_col].astype(int).values
    y_test = df_test[target_col].astype(int).values

    for c in cat_cols:
        X_train[c] = X_train[c].astype("category")
        X_test[c] = X_test[c].astype("category")

    return {
        "all_features": all_features,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def read_external_data(cfg: dict[str, Any]) -> pd.DataFrame:
    data_cfg = cfg["data"]
    return pd.read_excel(data_cfg["external_excel_path"], sheet_name=data_cfg["external_sheet"])


def build_marker_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["AGE", "AST", "ALT", "PLT", "BMI", "ALB", "M2BPGi", "DM_final", "Label"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if "FIB-4" not in out.columns:
        out["FIB-4"] = (out["AGE"] * out["AST"]) / (out["PLT"] * np.sqrt(out["ALT"]))

    if "NFS" not in out.columns:
        out["NFS"] = (
            -1.675
            + 0.037 * out["AGE"]
            + 0.094 * out["BMI"]
            + 1.13 * out["DM_final"]
            + 0.99 * (out["AST"] / out["ALT"])
            - 0.013 * out["PLT"]
            - 0.66 * out["ALB"]
        )
    return out
