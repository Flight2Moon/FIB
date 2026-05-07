from __future__ import annotations

import os
import platform
import subprocess

from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


@dataclass
class ModelBundle:
    name: str
    estimator: object


def _command_exists(cmd: str) -> bool:
    try:
        subprocess.run([cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False



def _has_nvidia_cuda() -> bool:
    if os.environ.get("FORCE_CPU", "0") == "1":
        return False
    if os.environ.get("FORCE_CUDA", "0") == "1":
        return True
    if platform.system() == "Darwin":
        return False
    if not _command_exists("nvidia-smi"):
        return False
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except Exception:
        return False



def _get_xgb_runtime_params() -> dict:
    # XGBoost supports CUDA, but not Apple Metal/MPS.
    if _has_nvidia_cuda():
        return {"device": "cuda", "tree_method": "hist"}
    return {"device": "cpu", "tree_method": "hist"}



def _get_catboost_runtime_params() -> dict:
    # CatBoost GPU backend requires NVIDIA CUDA and does not support Apple MPS.
    if _has_nvidia_cuda():
        return {"task_type": "GPU", "devices": "0"}
    return {"task_type": "CPU"}



def build_preprocessors(num_cols: list[str], cat_cols: list[str]):
    lr_pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols),
        ],
        remainder="drop",
    )

    tree_pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols),
        ],
        remainder="drop",
    )
    return lr_pre, tree_pre


def build_models(num_cols: list[str], cat_cols: list[str], scale_pos_weight: float, random_state: int, n_jobs: int, disable_catboost_files: bool = True, train_dir: str | None = None):
    lr_pre, tree_pre = build_preprocessors(num_cols, cat_cols)

    xgb_runtime = _get_xgb_runtime_params()
    cat_runtime = _get_catboost_runtime_params()

    lr = Pipeline([
        ("preprocess", lr_pre),
        ("model", LogisticRegression(
            class_weight="balanced",
            solver="liblinear",
            l1_ratio=0.0,
            max_iter=5000,
            random_state=random_state
        )),
    ])

    rf = Pipeline([
        ("preprocess", tree_pre),
        ("model", RandomForestClassifier(class_weight="balanced", random_state=random_state, n_jobs=n_jobs)),
    ])

    xgb = Pipeline([
        ("preprocess", tree_pre),
        ("model", XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            n_jobs=n_jobs,
            **xgb_runtime,
        )),
    ])

    cat_kwargs = dict(
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=random_state,
        auto_class_weights="Balanced",
        verbose=False,
    )
    if disable_catboost_files:
        cat_kwargs["allow_writing_files"] = False
    elif train_dir:
        cat_kwargs["train_dir"] = train_dir
    cat_kwargs.update(cat_runtime)
    cat = CatBoostClassifier(**cat_kwargs)

    return {
        "LogReg": lr,
        "RandomForest": rf,
        "XGBoost": xgb,
        "CatBoost": cat,
    }


def get_default_param_spaces(scale_pos_weight: float) -> dict[str, callable]:
    import optuna

    def xgb_param_fn(trial):
        lr = trial.suggest_float("model__learning_rate", 0.01, 0.1, log=True)
        return {
            "model__n_estimators": trial.suggest_int("model__n_estimators", 300, 2000, step=100),
            "model__max_depth": trial.suggest_int("model__max_depth", 2, 6),
            "model__learning_rate": lr,
            "model__subsample": trial.suggest_float("model__subsample", 0.6, 1.0),
            "model__colsample_bytree": trial.suggest_float("model__colsample_bytree", 0.5, 1.0),
            "model__min_child_weight": trial.suggest_float("model__min_child_weight", 1, 20, log=True),
            "model__gamma": trial.suggest_float("model__gamma", 0.0, 5.0),
            "model__reg_alpha": trial.suggest_float("model__reg_alpha", 1e-8, 10.0, log=True),
            "model__reg_lambda": trial.suggest_float("model__reg_lambda", 1e-3, 20.0, log=True),
            "model__scale_pos_weight": trial.suggest_float("model__scale_pos_weight", max(1e-3, scale_pos_weight * 0.5), scale_pos_weight * 1.5),
            **{f"model__{k}": v for k, v in _get_xgb_runtime_params().items()},
        }

    def rf_param_fn(trial):
        return {
            "model__n_estimators": trial.suggest_int("model__n_estimators", 300, 1200, step=100),
            "model__max_depth": trial.suggest_categorical("model__max_depth", [None, 3, 5, 7, 10, 15]),
            "model__min_samples_split": trial.suggest_int("model__min_samples_split", 2, 30),
            "model__min_samples_leaf": trial.suggest_int("model__min_samples_leaf", 1, 20),
            "model__max_features": trial.suggest_categorical("model__max_features", ["sqrt", "log2", None]),
            "model__max_samples": trial.suggest_float("model__max_samples", 0.6, 1.0),
        }

    def cat_param_fn(trial):
        return {
            "iterations": trial.suggest_int("iterations", 500, 3000, step=100),
            "depth": trial.suggest_int("depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
            "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
        }

    def lr_param_fn(trial):
        solver = trial.suggest_categorical("model__solver", ["liblinear", "saga"])

        params = {
            "model__solver": solver,
            "model__C": trial.suggest_float("model__C", 1e-4, 1e2, log=True),
        }

        if solver == "liblinear":
            l1_ratio_lib = trial.suggest_categorical("model__l1_ratio_liblinear", [0.0, 1.0])
            params["model__l1_ratio"] = l1_ratio_lib

        elif solver == "saga":
            l1_ratio_saga = trial.suggest_float("model__l1_ratio_saga", 0.0, 1.0)
            params["model__l1_ratio"] = l1_ratio_saga

        return params

    return {
        "LogReg": lr_param_fn,
        "RandomForest": rf_param_fn,
        "XGBoost": xgb_param_fn,
        "CatBoost": cat_param_fn,
    }
