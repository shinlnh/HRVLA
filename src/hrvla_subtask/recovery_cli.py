"""CLI for validation and high-throughput subtask-recovery evaluation."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

from .recovery import RECOVERY_METHODS, load_recovery_protocol
from .recovery_gpu_eval import evaluate_cosmos_recovery, write_cosmos_recovery_results
from .recovery_metrics import plot_recovery_summary, summarize_recovery, write_recovery_summary
from .recovery_simulator import recovery_episode_job, scenario_to_dict
from .simulator import load_suite, task_to_dict


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUITE = REPO_ROOT / "benchmark" / "subtask_suite_v1.json"
DEFAULT_PROTOCOL = REPO_ROOT / "benchmark" / "recovery_protocol_v1.json"


def _git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _gpu_name() -> str | None:
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _manifest(kind: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": kind,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "git_revision": _git_revision(),
        "python": sys.version,
        "platform": platform.platform(),
        "logical_cpu_count": os.cpu_count(),
        "gpu": _gpu_name(),
        "arguments": arguments,
    }


def command_validate(args: argparse.Namespace) -> int:
    tasks = load_suite(args.suite)
    scenarios = load_recovery_protocol(args.protocol, tasks)
    primitives = sum(len(item.recovery_primitives) for item in scenarios.values())
    print(f"valid recovery protocol: {len(tasks)} tasks, {primitives} recovery primitives")
    for task in tasks:
        scenario = scenarios[task.task_id]
        print(
            f"  {task.task_id}: {scenario.failure_label} -> "
            f"{len(scenario.recovery_primitives)} alternatives"
        )
    return 0


def _job_unpack(job: tuple[dict[str, Any], dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    return recovery_episode_job(*job)


def command_simulate(args: argparse.Namespace) -> int:
    tasks = load_suite(args.suite)
    scenarios = load_recovery_protocol(args.protocol, tasks)
    workers = args.workers or os.cpu_count() or 1
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for method in args.methods:
        for task_index, task in enumerate(tasks):
            scenario = scenarios[task.task_id]
            for episode in range(args.episodes_per_task):
                jobs.append(
                    (
                        task_to_dict(task),
                        scenario_to_dict(scenario),
                        {
                            "method": method,
                            "seed": args.seed + task_index * 1_000_000 + episode,
                            "disturbance_rate": args.disturbance_rate,
                            "proposal_error_rate": args.proposal_error_rate,
                            "max_extra_steps": args.max_extra_steps,
                        },
                    )
                )
    print(f"running {len(jobs)} paired recovery episodes with {workers} workers", flush=True)
    chunksize = max(1, len(jobs) // workers // 8)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(_job_unpack, jobs, chunksize=chunksize))
    records.sort(key=lambda row: (row["method"], row["task_id"], row["seed"]))

    raw_dir = output / "raw"
    raw_dir.mkdir(exist_ok=True)
    for method in args.methods:
        with gzip.open(raw_dir / f"{method}.jsonl.gz", "wt", encoding="utf-8") as stream:
            for record in records:
                if record["method"] == method:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
    summary = summarize_recovery(records)
    write_recovery_summary(summary, output)
    plot_recovery_summary(summary, output)
    manifest = _manifest(
        "symbolic_transition_recovery",
        {
            "suite": str(args.suite),
            "protocol": str(args.protocol),
            "methods": args.methods,
            "episodes_per_task": args.episodes_per_task,
            "workers": workers,
            "seed": args.seed,
            "disturbance_rate": args.disturbance_rate,
            "proposal_error_rate": args.proposal_error_rate,
            "max_extra_steps": args.max_extra_steps,
        },
    )
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def command_gpu_eval(args: argparse.Namespace) -> int:
    tasks = load_suite(args.suite)
    scenarios = load_recovery_protocol(args.protocol, tasks)
    records, summary = evaluate_cosmos_recovery(
        tasks,
        scenarios,
        model_path=args.model_path,
        candidates_per_failure=args.candidates_per_failure,
        repetitions=args.repetitions,
        seed=args.seed,
    )
    output = args.output_dir.resolve()
    write_cosmos_recovery_results(records, summary, output)
    manifest = _manifest(
        "cosmos_gpu_transition_recovery",
        {
            "suite": str(args.suite),
            "protocol": str(args.protocol),
            "model_path": str(args.model_path),
            "candidates_per_failure": args.candidates_per_failure,
            "repetitions": args.repetitions,
            "seed": args.seed,
            "shared_candidate_sets": True,
        },
    )
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, handler in (
        ("validate", command_validate),
        ("simulate", command_simulate),
        ("gpu-eval", command_gpu_eval),
    ):
        command = subparsers.add_parser(name)
        command.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
        command.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
        command.set_defaults(handler=handler)
        if name == "simulate":
            command.add_argument("--output-dir", type=Path, required=True)
            command.add_argument("--methods", nargs="+", default=list(RECOVERY_METHODS))
            command.add_argument("--episodes-per-task", type=int, default=200)
            command.add_argument("--workers", type=int, default=0)
            command.add_argument("--seed", type=int, default=42)
            command.add_argument("--disturbance-rate", type=float, default=0.15)
            command.add_argument("--proposal-error-rate", type=float, default=0.28)
            command.add_argument("--max-extra-steps", type=int, default=16)
        elif name == "gpu-eval":
            command.add_argument("--output-dir", type=Path, required=True)
            command.add_argument("--model-path", type=Path, required=True)
            command.add_argument("--candidates-per-failure", type=int, default=3)
            command.add_argument("--repetitions", type=int, default=1)
            command.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "episodes_per_task", 1) < 1:
        raise SystemExit("--episodes-per-task must be positive")
    if getattr(args, "workers", 0) < 0:
        raise SystemExit("--workers cannot be negative")
    if getattr(args, "repetitions", 1) < 1:
        raise SystemExit("--repetitions must be positive")
    for field in ("disturbance_rate", "proposal_error_rate"):
        value = getattr(args, field, 0.0)
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"--{field.replace('_', '-')} must be within [0, 1]")
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
