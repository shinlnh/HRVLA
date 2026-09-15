"""Predeclared statistical report for the five-row internal benchmark."""

from __future__ import annotations

from collections import defaultdict
import math
import statistics
from typing import Any, Iterable, Sequence

from .evidence import audit_evidence
from .internal_protocol import mcnemar_exact_two_sided, validate_internal_protocol_lock
from .plan import canonical_sha256
from .score import validate_record


def _rate(outcomes: Sequence[bool]) -> dict[str, Any]:
    count = len(outcomes)
    successes = sum(outcomes)
    if not count:
        return {"episodes": 0, "successes": 0, "rate": None, "wilson_95": None}
    rate = successes / count
    z = 1.959963984540054
    denominator = 1.0 + z * z / count
    center = (rate + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(rate * (1 - rate) / count + z * z / (4 * count**2)) / denominator
    return {
        "episodes": count,
        "successes": successes,
        "rate": rate,
        "wilson_95": [max(0.0, center - radius), min(1.0, center + radius)],
    }


def _mean_std(values: Sequence[float]) -> dict[str, float | None]:
    return {
        "mean": statistics.fmean(values) if values else None,
        "sample_std": statistics.stdev(values) if len(values) > 1 else None,
    }


def _nearest_rank(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))
    return float(ordered[index])


def _latency_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    policy = [
        float(value)
        for row in rows
        for value in row.get("policy_request_latencies_ms", [])
    ]
    planner = [
        float(value)
        for row in rows
        for value in row.get("planner_latencies_ms", [])
    ]
    wall = [float(row["wall_time_s"]) for row in rows]
    return {
        "policy_requests": len(policy),
        "policy_request_mean_ms": statistics.fmean(policy) if policy else None,
        "policy_request_p50_ms": _nearest_rank(policy, 0.50),
        "policy_request_p95_ms": _nearest_rank(policy, 0.95),
        "planner_calls": len(planner),
        "planner_mean_ms": statistics.fmean(planner) if planner else None,
        "planner_p95_ms": _nearest_rank(planner, 0.95),
        "episode_wall_mean_s": statistics.fmean(wall) if wall else None,
        "episode_wall_p95_s": _nearest_rank(wall, 0.95),
    }


def _slice_rates(
    rows: Sequence[dict[str, Any]], key_values: Iterable[tuple[str, str]]
) -> dict[str, dict[str, Any]]:
    output = {}
    for label, key in key_values:
        selected = [
            row["success"]
            for row in rows
            if row.get("failure") is not None
            and (
                row["failure"].get("level") == key
                or key in row["failure"].get("humanoid_axes", [])
            )
        ]
        output[label] = _rate(selected)
    return output


def method_report(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Compute method metrics while preserving protocol and training-seed units."""

    protocols = ("nominal", "failure_start", "online_failure")
    by_protocol = {name: [row for row in rows if row["protocol"] == name] for name in protocols}
    protocol_reports = {}
    for protocol, selected in by_protocol.items():
        cells: dict[tuple[str, str], list[bool]] = defaultdict(list)
        seeds: dict[int, list[bool]] = defaultdict(list)
        tasks: dict[str, list[bool]] = defaultdict(list)
        for row in selected:
            cells[(row["task_id"], row["scenario_id"])].append(row["success"])
            seeds[int(row["training_seed"])].append(row["success"])
            tasks[row["task_id"]].append(row["success"])
        cell_rates = [sum(values) / len(values) for values in cells.values()]
        seed_rates = {
            str(seed): sum(values) / len(values) for seed, values in sorted(seeds.items())
        }
        successful_recovery_times = [
            float(row["recovery_time_s"])
            for row in selected
            if row["success"] and row.get("recovery_time_s") is not None
        ]
        protocol_reports[protocol] = {
            "success": _rate([row["success"] for row in selected]),
            "macro_cell_success_rate": statistics.fmean(cell_rates) if cell_rates else None,
            "cells": len(cells),
            "by_task": {task: _rate(values) for task, values in sorted(tasks.items())},
            "by_training_seed": seed_rates,
            "training_seed_dispersion": _mean_std(list(seed_rates.values())),
            "fall_rate": sum(row["termination"] == "fall" for row in selected) / len(selected)
            if selected
            else None,
            "safety_violation_rate": sum(row.get("safety_violations", 0) > 0 for row in selected)
            / len(selected)
            if selected
            else None,
            "mean_successful_recovery_time_s": (
                statistics.fmean(successful_recovery_times)
                if successful_recovery_times
                else None
            ),
            "by_level_and_axis": _slice_rates(
                selected,
                [(value, value) for value in ("L1", "L2", "L3", "L4", "H1", "H2", "H3")],
            ),
            "efficiency": _latency_summary(selected),
        }

    nominal_rate = protocol_reports["nominal"]["success"]["rate"]
    online_rate = protocol_reports["online_failure"]["success"]["rate"]
    recovery_tasks: dict[str, list[bool]] = defaultdict(list)
    for row in by_protocol["online_failure"]:
        recovery_tasks[row["task_id"]].append(row["success"])
    task_rates = [sum(values) / len(values) for values in recovery_tasks.values()]
    detection_rows = [
        row
        for row in [*by_protocol["nominal"], *by_protocol["online_failure"]]
        if type(row.get("detected")) is bool
    ]
    tp = sum(row["protocol"] == "online_failure" and row["detected"] for row in detection_rows)
    fp = sum(row["protocol"] == "nominal" and row["detected"] for row in detection_rows)
    fn = sum(row["protocol"] == "online_failure" and not row["detected"] for row in detection_rows)
    tn = sum(row["protocol"] == "nominal" and not row["detected"] for row in detection_rows)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {
        "protocols": protocol_reports,
        "recovery_degradation_nominal_minus_online": (
            nominal_rate - online_rate
            if nominal_rate is not None and online_rate is not None
            else None
        ),
        "recovery_consistency_one_minus_task_rate_sd": (
            1.0 - statistics.pstdev(task_rates) if task_rates else None
        ),
        "detection": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": (
                2 * precision * recall / (precision + recall)
                if precision is not None and recall is not None and precision + recall
                else None
            ),
            "mean_true_positive_latency_s": statistics.fmean(
                float(row["detection_latency_s"])
                for row in detection_rows
                if row["protocol"] == "online_failure"
                and row["detected"]
                and row.get("detection_latency_s") is not None
            )
            if tp
            else None,
        },
    }


def _pair_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["task_id"],
        row["scenario_id"],
        row["protocol"],
        row["training_seed"],
        row["rollout_seed"],
        row["initial_snapshot_id"],
        row.get("failure_snapshot_id"),
    )


def primary_comparisons(
    records: Sequence[dict[str, Any]],
    lock: dict[str, Any],
    *,
    expected_pairs_per_cell: int | None = None,
) -> list[dict[str, Any]]:
    """Run exactly the four frozen paired tests, then apply Holm correction."""

    primary_tasks = set(lock["analysis"]["primary_task_aggregate"])
    output = []
    for endpoint in lock["analysis"]["primary_comparison_endpoints"]:
        protocol = endpoint["protocol"]
        candidate = {
            _pair_key(row): bool(row["success"])
            for row in records
            if row["method_id"] == endpoint["candidate"]
            and row["protocol"] == protocol
            and row["task_id"] in primary_tasks
        }
        baseline = {
            _pair_key(row): bool(row["success"])
            for row in records
            if row["method_id"] == endpoint["baseline"]
            and row["protocol"] == protocol
            and row["task_id"] in primary_tasks
        }
        if not candidate or set(candidate) != set(baseline):
            raise ValueError(f"{endpoint['family']}: primary paired evidence is incomplete")
        cell_counts: dict[tuple[str, str], int] = defaultdict(int)
        for key in candidate:
            cell_counts[(str(key[0]), str(key[1]))] += 1
        if expected_pairs_per_cell is not None and any(
            count != expected_pairs_per_cell for count in cell_counts.values()
        ):
            raise ValueError(f"{endpoint['family']}: primary cell replication differs")
        pairs = [(candidate[key], baseline[key], key) for key in sorted(candidate)]
        candidate_only = sum(left and not right for left, right, _key in pairs)
        baseline_only = sum(right and not left for left, right, _key in pairs)
        p_value = mcnemar_exact_two_sided(candidate_only, baseline_only)
        seed_deltas = {}
        for seed in sorted({int(key[3]) for key in candidate}):
            selected = [(left, right) for left, right, key in pairs if int(key[3]) == seed]
            seed_deltas[str(seed)] = statistics.fmean(
                float(left) - float(right) for left, right in selected
            )
        output.append(
            {
                **endpoint,
                "paired_episodes": len(pairs),
                "paired_cells": len(cell_counts),
                "pairs_per_cell": sorted(set(cell_counts.values())),
                "candidate_success_rate": statistics.fmean(left for left, _right, _key in pairs),
                "baseline_success_rate": statistics.fmean(right for _left, right, _key in pairs),
                "paired_risk_difference": statistics.fmean(
                    float(left) - float(right) for left, right, _key in pairs
                ),
                "matched_odds_ratio_haldane": (candidate_only + 0.5) / (baseline_only + 0.5),
                "candidate_only_success": candidate_only,
                "baseline_only_success": baseline_only,
                "ties": len(pairs) - candidate_only - baseline_only,
                "mcnemar_exact_two_sided_p": p_value,
                "training_seed_risk_differences": seed_deltas,
                "training_seed_effect_dispersion": _mean_std(list(seed_deltas.values())),
            }
        )

    ranked = sorted(
        range(len(output)),
        key=lambda index: output[index]["mcnemar_exact_two_sided_p"],
    )
    running = 0.0
    total = len(ranked)
    for rank, index in enumerate(ranked):
        adjusted = min(1.0, output[index]["mcnemar_exact_two_sided_p"] * (total - rank))
        running = max(running, adjusted)
        output[index]["holm_adjusted_p"] = running
        output[index]["reject_familywise_0_05"] = running <= 0.05
    return output


def hardware_report(samples: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    if not samples:
        return None
    fields = (
        "cpu_busy_percent",
        "gpu_utilization_percent",
        "gpu_memory_used_mib",
        "gpu_power_w",
        "gpu_temperature_c",
        "memory_available_gib",
    )
    return {
        "samples": len(samples),
        **{
            field: {
                "mean": statistics.fmean(float(row[field]) for row in samples),
                "p95": _nearest_rank([float(row[field]) for row in samples], 0.95),
                "maximum": max(float(row[field]) for row in samples),
            }
            for field in fields
        },
    }


def build_internal_report(
    plan: dict[str, Any],
    records: Iterable[dict[str, Any]],
    lock: dict[str, Any],
    telemetry: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Validate complete hidden evidence and build the claim-bounded final report."""

    validate_internal_protocol_lock(lock)
    if lock["splits"]["hidden_final"]["claim_eligible"] is not True:
        raise ValueError("hidden_final is not claim eligible")
    rows = list(records)
    for row in rows:
        validate_record(row)
    evidence = audit_evidence(plan, rows)
    primary_tasks = set(lock["analysis"]["primary_task_aggregate"])
    diagnostic_tasks = set(lock["analysis"]["diagnostic_only_tasks"])
    primary = [row for row in rows if row["task_id"] in primary_tasks]
    diagnostic = [row for row in rows if row["task_id"] in diagnostic_tasks]
    methods = list(lock["methods"])
    expected_pairs = (
        len(lock["training_seeds"])
        * int(lock["splits"]["hidden_final"]["rollouts_per_training_seed_per_cell"])
    )
    core = {
        "schema_version": 1,
        "claim_status": "hidden_final_complete",
        "claim_boundary": (
            "Primary aggregate excludes locked OpenDoor diagnostics. Development and validation "
            "results are excluded. Four predeclared paired tests alone receive Holm correction."
        ),
        "plan_sha256": plan["plan_sha256"],
        "suite_sha256": plan["suite_sha256"],
        "evidence_audit": evidence,
        "episode_records": len(rows),
        "primary_episode_records": len(primary),
        "diagnostic_episode_records": len(diagnostic),
        "methods": {
            method: method_report([row for row in primary if row["method_id"] == method])
            for method in methods
        },
        "open_door_diagnostic": {
            method: method_report([row for row in diagnostic if row["method_id"] == method])
            for method in methods
        },
        "primary_comparisons": primary_comparisons(
            primary, lock, expected_pairs_per_cell=expected_pairs
        ),
        "power_analysis": lock["power_analysis"],
        "hardware": hardware_report(telemetry),
    }
    return {**core, "report_sha256": canonical_sha256(core)}


__all__ = [
    "build_internal_report",
    "hardware_report",
    "method_report",
    "primary_comparisons",
]
