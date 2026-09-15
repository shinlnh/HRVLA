#!/usr/bin/env python3
"""Compile the immutable validation decision required before hidden-final access."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hrvla_bench.hidden_final_gate import build_hidden_final_gate  # noqa: E402
from hrvla_bench.plan import load_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans-root", type=Path, default=ROOT / "_artifacts/HumanoidArena/internal-benchmark/plans")
    parser.add_argument("--validation-root", type=Path, default=ROOT / "_artifacts/HumanoidArena/internal-benchmark/runs/validation")
    parser.add_argument("--protocol-lock", type=Path, default=ROOT / "config/humanoidarena-internal-protocol.lock.json")
    parser.add_argument("--checkpoint-lock", type=Path, default=ROOT / "config/humanoidarena-internal-checkpoints.lock.json")
    parser.add_argument("--method-programs", type=Path, default=ROOT / "benchmark/humanoidarena_method_programs.json")
    parser.add_argument("--output", type=Path, default=ROOT / "_artifacts/HumanoidArena/internal-benchmark/hidden-final-gate.candidate.json")
    args = parser.parse_args()
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    gate = build_hidden_final_gate(
        validation_plan=load_json(args.plans_root.resolve() / "validation.plan.json"),
        protocol=load_json(args.protocol_lock.resolve()),
        checkpoint_lock=load_json(args.checkpoint_lock.resolve()),
        method_programs=load_json(args.method_programs.resolve()),
        validation_root=args.validation_root.resolve(),
        source_revision_before_freeze=revision,
    )
    if args.output.exists():
        raise FileExistsError(f"refusing to replace hidden-final candidate gate: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({"gate_sha256": gate["gate_sha256"], "methods": len(gate["methods"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
