"""Dependency-free validation and scoring for HRVLA episode JSONL records."""

from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Sequence


PROTOCOLS = {"nominal", "failure_start", "online_failure"}
LEVELS = {"L1", "L2", "L3", "L4"}
AXES = {"H1", "H2", "H3"}
SEVERITIES = {"low", "medium", "high"}
TERMINATIONS = {"success", "timeout", "fall", "safety", "unrecoverable", "other"}
REQUIRED = {
    "schema_version",
    "run_id",
    "episode_id",
    "method_id",
    "task_id",
    "scenario_id",
    "protocol",
    "training_seed",
    "rollout_seed",
    "initial_snapshot_id",
    "policy_checkpoint_id",
    "controller_id",
    "simulator_revision",
    "success",
    "termination",
}


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def validate_record(record: dict[str, Any]) -> None:
    """Raise ValueError when an episode violates the benchmark contract."""
    missing = sorted(REQUIRED - set(record))
    if missing:
        raise ValueError(f"missing required field(s): {', '.join(missing)}")
    if record["schema_version"] != "1.0":
        raise ValueError("schema_version must be '1.0'")
    if record["protocol"] not in PROTOCOLS:
        raise ValueError(f"unknown protocol: {record['protocol']!r}")
    if type(record["success"]) is not bool:
        raise ValueError("success must be boolean")
    if record["termination"] not in TERMINATIONS:
        raise ValueError(f"unknown termination: {record['termination']!r}")
    if record["success"] != (record["termination"] == "success"):
        raise ValueError("success must agree with termination == 'success'")
    for field in ("training_seed", "rollout_seed"):
        if not _is_nonnegative_int(record[field]):
            raise ValueError(f"{field} must be a non-negative integer")

    protocol = record["protocol"]
    failure = record.get("failure")
    if protocol == "nominal":
        if failure is not None:
            raise ValueError("nominal episodes must not contain a failure")
    else:
        if not isinstance(failure, dict):
            raise ValueError("recovery episodes require a failure object")
        if failure.get("level") not in LEVELS:
            raise ValueError("failure.level must be L1, L2, L3, or L4")
        if failure.get("severity") not in SEVERITIES:
            raise ValueError("failure.severity must be low, medium, or high")
        axes = failure.get("humanoid_axes")
        if not isinstance(axes, list) or len(axes) != len(set(axes)):
            raise ValueError("failure.humanoid_axes must be a unique list")
        if not set(axes).issubset(AXES):
            raise ValueError("failure.humanoid_axes contains an unknown axis")
        if failure.get("recoverable_oracle") is not True:
            raise ValueError("a recovery scenario must be oracle-verified recoverable")
        if not failure.get("event_id") or not failure.get("event_boundary"):
            raise ValueError("failure event_id and event_boundary are required")
        if protocol == "failure_start" and not record.get("failure_snapshot_id"):
            raise ValueError("failure_start requires failure_snapshot_id")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("record must be a JSON object")
                validate_record(record)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            records.append(record)
    if not records:
        raise ValueError(f"{path}: no episode records")
    return records


def _proportion(values: Sequence[bool]) -> dict[str, Any]:
    n = len(values)
    successes = sum(values)
    if not n:
        return {"episodes": 0, "successes": 0, "rate": None, "wilson_95": None}
    rate = successes / n
    z = 1.959963984540054
    denominator = 1 + z * z / n
    center = (rate + z * z / (2 * n)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / denominator
    return {
        "episodes": n,
        "successes": successes,
        "rate": rate,
        "wilson_95": [max(0.0, center - half), min(1.0, center + half)],
    }


def _recovery_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    micro = _proportion([record["success"] for record in records])
    cells: dict[tuple[Any, ...], list[bool]] = defaultdict(list)
    levels: dict[str, list[bool]] = defaultdict(list)
    axes: dict[str, list[bool]] = defaultdict(list)
    seeds: dict[int, list[bool]] = defaultdict(list)
    for record in records:
        failure = record["failure"]
        axis_key = tuple(sorted(failure["humanoid_axes"])) or ("none",)
        cell = (record["task_id"], failure["level"], axis_key, failure["severity"])
        cells[cell].append(record["success"])
        levels[failure["level"]].append(record["success"])
        for axis in axis_key:
            axes[axis].append(record["success"])
        seeds[record["training_seed"]].append(record["success"])

    cell_rates = [sum(values) / len(values) for values in cells.values()]
    seed_rates = {str(seed): sum(values) / len(values) for seed, values in sorted(seeds.items())}
    seed_values = list(seed_rates.values())
    times = [
        float(record["recovery_time_s"])
        for record in records
        if record["success"] and "recovery_time_s" in record
    ]
    degradation = [
        1 - (float(record["post_failure_score"]) + 1e-8)
        / (float(record["pre_failure_score"]) + 1e-8)
        for record in records
        if "pre_failure_score" in record and "post_failure_score" in record
    ]
    return {
        "micro_rsr": micro,
        "macro_rsr": statistics.fmean(cell_rates) if cell_rates else None,
        "macro_cells": len(cell_rates),
        "by_level": {key: _proportion(levels[key]) for key in sorted(levels)},
        "by_humanoid_axis": {key: _proportion(axes[key]) for key in sorted(axes)},
        "by_training_seed": seed_rates,
        "seed_mean_rsr": statistics.fmean(seed_values) if seed_values else None,
        "seed_sample_std_rsr": statistics.stdev(seed_values) if len(seed_values) > 1 else None,
        "fall_rate": sum(record["termination"] == "fall" for record in records) / len(records),
        "safety_violation_rate": sum(
            record.get("safety_violations", 0) > 0 for record in records
        )
        / len(records),
        "mean_successful_recovery_time_s": statistics.fmean(times) if times else None,
        "mean_recovery_degradation": statistics.fmean(degradation) if degradation else None,
        "recovery_consistency": _recovery_consistency(records),
    }


def _detection_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [
        record
        for record in records
        if record["protocol"] in {"nominal", "online_failure"}
        and type(record.get("detected")) is bool
    ]
    if not eligible:
        return None
    tp = sum(record["protocol"] == "online_failure" and record["detected"] for record in eligible)
    fp = sum(record["protocol"] == "nominal" and record["detected"] for record in eligible)
    fn = sum(record["protocol"] == "online_failure" and not record["detected"] for record in eligible)
    tn = sum(record["protocol"] == "nominal" and not record["detected"] for record in eligible)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    latencies = [
        float(record["detection_latency_s"])
        for record in eligible
        if record["protocol"] == "online_failure"
        and record["detected"]
        and "detection_latency_s" in record
    ]
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_true_positive_latency_s": statistics.fmean(latencies) if latencies else None,
    }


def _recovery_consistency(records: Sequence[dict[str, Any]]) -> float | None:
    tasks: dict[str, list[bool]] = defaultdict(list)
    for record in records:
        if record["protocol"] != "nominal":
            tasks[record["task_id"]].append(record["success"])
    rates = [sum(values) / len(values) for values in tasks.values()]
    return 1 - statistics.pstdev(rates) if rates else None


def _mcnemar_exact(a_wins: int, b_wins: int) -> float | None:
    discordant = a_wins + b_wins
    if not discordant:
        return None
    tail = sum(math.comb(discordant, k) for k in range(min(a_wins, b_wins) + 1))
    return min(1.0, 2 * tail / (2**discordant))


def _paired_comparisons(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_protocol_method: dict[str, dict[str, dict[tuple[Any, ...], bool]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for record in records:
        if record["protocol"] == "nominal":
            continue
        key = (
            record["task_id"],
            record["scenario_id"],
            record["training_seed"],
            record["rollout_seed"],
        )
        method_records = by_protocol_method[record["protocol"]][record["method_id"]]
        if key in method_records:
            raise ValueError(
                f"duplicate paired comparison key for {record['protocol']}/"
                f"{record['method_id']}: {key}"
            )
        method_records[key] = record["success"]

    output: list[dict[str, Any]] = []
    for protocol, by_method in sorted(by_protocol_method.items()):
        for method_a, method_b in combinations(sorted(by_method), 2):
            common = sorted(set(by_method[method_a]) & set(by_method[method_b]))
            if not common:
                continue
            outcomes = [
                (by_method[method_a][key], by_method[method_b][key]) for key in common
            ]
            a_wins = sum(a and not b for a, b in outcomes)
            b_wins = sum(b and not a for a, b in outcomes)
            ties = len(outcomes) - a_wins - b_wins
            output.append(
                {
                    "protocol": protocol,
                    "method_a": method_a,
                    "method_b": method_b,
                    "paired_episodes": len(outcomes),
                    "success_rate_delta_a_minus_b": statistics.fmean(
                        float(a) - float(b) for a, b in outcomes
                    ),
                    "a_only_success": a_wins,
                    "b_only_success": b_wins,
                    "ties": ties,
                    "mcnemar_exact_two_sided_p": _mcnemar_exact(a_wins, b_wins),
                }
            )
    return output


def score_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    materialized = list(records)
    if not materialized:
        raise ValueError("no records to score")
    for record in materialized:
        validate_record(record)

    methods: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in materialized:
        methods[record["method_id"]].append(record)

    method_reports: dict[str, Any] = {}
    for method, method_records in sorted(methods.items()):
        nominal = [record for record in method_records if record["protocol"] == "nominal"]
        failure_start = [
            record for record in method_records if record["protocol"] == "failure_start"
        ]
        online = [record for record in method_records if record["protocol"] == "online_failure"]
        method_reports[method] = {
            "episodes": len(method_records),
            "nominal_success": _proportion([record["success"] for record in nominal]),
            "failure_start": _recovery_summary(failure_start) if failure_start else None,
            "online_failure": _recovery_summary(online) if online else None,
            "detection": _detection_summary(method_records),
        }

    return {
        "schema_version": "1.0",
        "episode_records": len(materialized),
        "methods": method_reports,
        "paired_comparisons": _paired_comparisons(materialized),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", type=Path, help="Episode records in JSONL format")
    parser.add_argument("--output", type=Path, help="Write summary JSON to this path")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = score_records(load_jsonl(args.episodes))
    except (OSError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
