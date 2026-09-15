from __future__ import annotations

import json
from pathlib import Path

import pytest

from hrvla_bench.internal_protocol import validate_internal_protocol_lock
from hrvla_bench.internal_report import hardware_report, method_report, primary_comparisons


ROOT = Path(__file__).resolve().parents[1]
LOCK = json.loads(
    (ROOT / "config/humanoidarena-internal-protocol.lock.json").read_text(encoding="utf-8")
)


def _row(method: str, protocol: str, rollout: int, success: bool) -> dict:
    return {
        "method_id": method,
        "protocol": protocol,
        "task_id": "boxing",
        "scenario_id": "nominal" if protocol == "nominal" else "contact-recoil",
        "training_seed": rollout % 2,
        "rollout_seed": rollout,
        "initial_snapshot_id": "initial",
        "failure_snapshot_id": None,
        "success": success,
        "termination": "success" if success else "timeout",
        "safety_violations": 0,
        "wall_time_s": 10.0 + rollout,
        "policy_request_latencies_ms": [10.0, 20.0],
        "planner_latencies_ms": [0.5],
        "failure": None
        if protocol == "nominal"
        else {"level": "L2", "humanoid_axes": ["H1"]},
    }


def test_four_primary_comparisons_use_only_frozen_endpoints_and_holm() -> None:
    validate_internal_protocol_lock(LOCK)
    records = []
    outcomes = {
        "gr00t_sonic": {"nominal": [False, False, True, False]},
        "gr00t_st": {
            "nominal": [True, False, True, True],
            "online_failure": [False, False, True, False],
        },
        "gr00t_st_rt": {"nominal": [True, True, True, True]},
        "gr00t_str": {"online_failure": [True, False, True, True]},
        "gr00t_str_rt": {"online_failure": [True, True, True, True]},
    }
    for method, protocols in outcomes.items():
        for protocol, values in protocols.items():
            records.extend(
                _row(method, protocol, index, value) for index, value in enumerate(values)
            )
    comparisons = primary_comparisons(records, LOCK, expected_pairs_per_cell=4)
    assert [row["family"] for row in comparisons] == LOCK["power_analysis"][
        "primary_comparison_families"
    ]
    assert all(row["paired_episodes"] == 4 for row in comparisons)
    assert all(0.0 <= row["holm_adjusted_p"] <= 1.0 for row in comparisons)
    assert comparisons[0]["paired_risk_difference"] == pytest.approx(0.5)


def test_method_and_hardware_reports_keep_efficiency_and_robustness_slices() -> None:
    rows = [
        _row("gr00t_str", "online_failure", 0, True),
        _row("gr00t_str", "online_failure", 1, False),
    ]
    report = method_report(rows)
    online = report["protocols"]["online_failure"]
    assert online["success"]["rate"] == 0.5
    assert online["by_level_and_axis"]["L2"]["episodes"] == 2
    assert online["by_level_and_axis"]["H1"]["episodes"] == 2
    assert online["efficiency"]["policy_requests"] == 4
    hardware = hardware_report(
        [
            {
                "cpu_busy_percent": 80,
                "gpu_utilization_percent": 90,
                "gpu_memory_used_mib": 12000,
                "gpu_power_w": 180,
                "gpu_temperature_c": 65,
                "memory_available_gib": 8,
            },
            {
                "cpu_busy_percent": 100,
                "gpu_utilization_percent": 100,
                "gpu_memory_used_mib": 14000,
                "gpu_power_w": 220,
                "gpu_temperature_c": 70,
                "memory_available_gib": 6,
            },
        ]
    )
    assert hardware is not None
    assert hardware["cpu_busy_percent"]["mean"] == 90
    assert hardware["gpu_memory_used_mib"]["maximum"] == 14000
