#!/usr/bin/env python3
"""Render a cross-disturbance comparison from retained recovery result cells."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hrvla_subtask.recovery_report import aggregate_recovery_cells  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cells", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate_recovery_cells(args.cells, args.output_dir)
    rates = [cell["disturbance_rate"] for cell in result["cells"]]
    print(f"rendered {len(rates)} recovery cells at disturbance rates {rates}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
