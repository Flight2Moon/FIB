from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    return cfg


def ensure_run_dir(cfg: dict[str, Any]) -> Path:
    out_root = Path(cfg["output"]["out_root"])
    run_name = cfg["output"].get("run_name")
    if not run_name:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = stamp
        cfg["output"]["run_name"] = run_name

    run_dir = out_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg["output"]["run_dir"] = str(run_dir)
    return run_dir
