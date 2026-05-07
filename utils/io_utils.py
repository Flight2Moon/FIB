from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import joblib
import pandas as pd


def save_json(obj: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")



def save_excel(sheets: Mapping[str, pd.DataFrame], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)



def save_bundle(bundle: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, str(path))



def load_bundle(path: str | Path) -> dict:
    return joblib.load(str(path))



def find_latest_result_dir(results_root: str | Path) -> Path:
    root = Path(results_root)

    if not root.exists():
        raise FileNotFoundError(f"results root not found: {root}")

    required_bundle_files = {
        "logreg_bundle.joblib",
        "randomforest_bundle.joblib",
        "xgboost_bundle.joblib",
        "catboost_bundle.joblib",
    }

    root_files = {p.name for p in root.iterdir() if p.is_file()}
    if required_bundle_files.issubset(root_files):
        return root

    candidates = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        child_files = {x.name for x in p.iterdir() if x.is_file()}
        if required_bundle_files.issubset(child_files):
            candidates.append(p)

    if not candidates:
        existing = sorted(x.name for x in root.iterdir())
        raise FileNotFoundError(
            f"no valid run directories with all bundle files found under: {root}. "
            f"existing entries: {existing}"
        )

    latest_dir = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest_dir
