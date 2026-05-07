from __future__ import annotations

import argparse
from pathlib import Path

from utils.config import load_config, ensure_run_dir
from pipelines.internal import run_internal_pipeline
from pipelines.external import run_external_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FIB modular pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_int = sub.add_parser("internal", help="Run internal training/evaluation")
    p_int.add_argument("--config", required=True, help="Path to internal YAML config")

    p_ext = sub.add_parser("external", help="Run external validation")
    p_ext.add_argument("--config", required=True, help="Path to external YAML config")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_run_dir(cfg)

    if args.command == "internal":
        run_internal_pipeline(cfg)
    elif args.command == "external":
        run_external_pipeline(cfg)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
