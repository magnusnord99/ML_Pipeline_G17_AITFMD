"""Entry point for preprocessing pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.index_dataset import build_dataset_index, save_dataset_index, summarize_index


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _resolve_from_config(config_path: Path, raw_path: str) -> Path:
    """Resolve project paths relative to config file location."""
    return (config_path.parent / raw_path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run preprocessing steps.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/preprocessing.yaml",
        help="Path to preprocessing YAML config.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = _load_yaml(config_path)

    dataset_root = _resolve_from_config(config_path, cfg["paths"]["input_dataset_root"])
    metadata_csv = _resolve_from_config(config_path, cfg["paths"]["metadata_csv"])
    tumor_suffix = cfg["labels"]["tumor_suffix"]
    non_tumor_suffix = cfg["labels"]["non_tumor_suffix"]

    print(f"[index] dataset root: {dataset_root}")
    df = build_dataset_index(
        dataset_root=dataset_root,
        tumor_suffix=tumor_suffix,
        non_tumor_suffix=non_tumor_suffix,
    )
    out_path = save_dataset_index(df, metadata_csv)
    summary = summarize_index(df)

    print(f"[index] wrote metadata CSV: {out_path}")
    print("[index] summary:")
    for key, value in summary.items():
        print(f"  - {key}: {value}")


if __name__ == "__main__":
    main()
