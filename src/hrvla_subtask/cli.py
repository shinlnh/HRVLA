"""Command-line interface for validation, CPU simulation, and GPU evaluation."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

from .gpu_eval import evaluate_cosmos, write_gpu_results
from .metrics import plot_summary, summarize, write_chart_data, write_summary
from .simulator import episode_job, load_suite, task_to_dict


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUITE = REPO_ROOT / "benchmark" / "subtask_suite_v1.json"
DEFAULT_METHODS = ["plan_once", "recursive", "best_of_n", "ttc", "adaptive_ttc"]


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
    print(f"valid suite: {len(tasks)} tasks, {sum(len(task.skills) for task in tasks)} skills")
    for task in tasks:
        print(f"  {task.task_id}: {len(task.skills)} skills, max_steps={task.max_steps}")
    return 0


def command_simulate(args: argparse.Namespace) -> int:
    tasks = load_suite(args.suite)
    workers = args.workers or os.cpu_count() or 1
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for method in args.methods:
        for task_index, task in enumerate(tasks):
            raw_task = task_to_dict(task)
            for episode in range(args.episodes_per_task):
                jobs.append(
                    (
                        raw_task,
                        {
                            "method": method,
                            "seed": args.seed + task_index * 1_000_000 + episode,
                            "proposal_error_rate": args.proposal_error_rate,
                            "planner_overrides": {
                                "branching_factor": args.branching_factor,
                                "beam_width": args.beam_width,
                                "search_depth": args.search_depth,
                                "confidence_threshold": args.confidence_threshold,
                            },
                        },
                    )
                )
    print(f"running {len(jobs)} episodes with {workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(_episode_job_unpack, jobs, chunksize=max(1, len(jobs) // workers // 8)))
    records.sort(key=lambda row: (row["method"], row["task_id"], row["seed"]))

    raw_dir = output / "raw"
    raw_dir.mkdir(exist_ok=True)
    for method in args.methods:
        path = raw_dir / f"{method}.jsonl"
        path.write_text(
            "".join(
                json.dumps(record, sort_keys=True) + "\n"
                for record in records
                if record["method"] == method
            ),
            encoding="utf-8",
        )
    summary = summarize(records)
    write_summary(summary, output / "summary.json")
    write_chart_data(summary, output / "chart_data.csv")
    plot_summary(summary, output)
    manifest = _manifest(
        "symbolic_cpu_simulation",
        {
            "suite": str(args.suite),
            "methods": args.methods,
            "episodes_per_task": args.episodes_per_task,
            "workers": workers,
            "seed": args.seed,
            "proposal_error_rate": args.proposal_error_rate,
            "branching_factor": args.branching_factor,
            "beam_width": args.beam_width,
            "search_depth": args.search_depth,
            "confidence_threshold": args.confidence_threshold,
        },
    )
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _episode_job_unpack(job: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    return episode_job(*job)


def command_gpu_eval(args: argparse.Namespace) -> int:
    tasks = load_suite(args.suite)
    records, summary = evaluate_cosmos(
        tasks,
        model_path=args.model_path,
        methods=args.methods,
        max_points_per_task=args.max_points_per_task,
        seed=args.seed,
        config_overrides={
            "branching_factor": args.branching_factor,
            "beam_width": args.beam_width,
            "search_depth": args.search_depth,
            "confidence_threshold": args.confidence_threshold,
        },
    )
    output = args.output_dir.resolve()
    write_gpu_results(records, summary, output)
    manifest = _manifest(
        "cosmos_gpu_next_subtask",
        {
            "suite": str(args.suite),
            "model_path": str(args.model_path),
            "methods": args.methods,
            "max_points_per_task": args.max_points_per_task,
            "seed": args.seed,
            "branching_factor": args.branching_factor,
            "beam_width": args.beam_width,
            "search_depth": args.search_depth,
            "confidence_threshold": args.confidence_threshold,
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

    validate = subparsers.add_parser("validate", help="validate the task/skill suite")
    validate.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    validate.set_defaults(handler=command_validate)

    simulate = subparsers.add_parser(
        "simulate", help="run parallel symbolic closed-loop rollouts"
    )
    simulate.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    simulate.add_argument("--output-dir", type=Path, required=True)
    simulate.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    simulate.add_argument("--episodes-per-task", type=int, default=200)
    simulate.add_argument("--workers", type=int, default=0, help="0 uses every logical CPU")
    simulate.add_argument("--seed", type=int, default=42)
    simulate.add_argument("--proposal-error-rate", type=float, default=0.28)
    _add_planner_arguments(simulate)
    simulate.set_defaults(handler=command_simulate)

    gpu = subparsers.add_parser("gpu-eval", help="evaluate real Cosmos-Reason2 proposals")
    gpu.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    gpu.add_argument("--output-dir", type=Path, required=True)
    gpu.add_argument("--model-path", type=Path, required=True)
    gpu.add_argument(
        "--methods",
        nargs="+",
        default=["plan_once", "best_of_n", "adaptive_ttc"],
    )
    gpu.add_argument("--max-points-per-task", type=int, default=3)
    gpu.add_argument("--seed", type=int, default=42)
    _add_planner_arguments(gpu)
    gpu.set_defaults(handler=command_gpu_eval)
    return parser


def _add_planner_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--branching-factor", type=int, default=4)
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--search-depth", type=int, default=3)
    parser.add_argument("--confidence-threshold", type=float, default=0.72)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "episodes_per_task", 1) < 1:
        raise SystemExit("--episodes-per-task must be positive")
    if getattr(args, "workers", 0) < 0:
        raise SystemExit("--workers cannot be negative")
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
