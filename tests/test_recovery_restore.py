from __future__ import annotations

from pathlib import Path
import json
import types

import pytest

from hrvla_bench.isaac_snapshot import (
    capture_scene_snapshot,
    snapshot_state_sha256,
    write_snapshot_atomic,
)
from hrvla_bench.plan import canonical_sha256
from hrvla_bench.recovery_restore import (
    audit_failure_start_trial,
    restore_snapshot_for_trial,
)


ROOT = Path(__file__).resolve().parents[1]
SUITE = json.loads(
    (ROOT / "benchmark/suites/hrvla_recovery_v0.json").read_text(encoding="utf-8")
)


class _Asset:
    def __init__(self, torch_module, x: float) -> None:
        self.device = torch_module.device("cpu")
        root = torch_module.zeros((1, 13), dtype=torch_module.float32)
        root[:, 0] = x
        root[:, 2] = 1.0
        root[:, 3] = 1.0
        self.data = types.SimpleNamespace(root_state_w=root)
        self.joint_names = []

    def write_root_state_to_sim(self, value, *, env_ids) -> None:
        self.data.root_state_w[env_ids] = value


class _Scene(dict):
    def __init__(self, torch_module, x: float) -> None:
        super().__init__(robot=_Asset(torch_module, x), box=_Asset(torch_module, x + 1.0))
        self.write_count = 0

    def write_data_to_sim(self) -> None:
        self.write_count += 1


class _Simulation:
    def __init__(self) -> None:
        self.forward_count = 0

    def forward(self) -> None:
        self.forward_count += 1

    def has_rtx_sensors(self) -> bool:
        return False


class _Env:
    def __init__(self, torch_module, x: float, step: int) -> None:
        self.num_envs = 1
        self.device = torch_module.device("cpu")
        self.scene = _Scene(torch_module, x)
        self.episode_length_buf = torch_module.tensor([step], dtype=torch_module.long)
        self.sim = _Simulation()
        self.cfg = types.SimpleNamespace(rerender_on_reset=False)


def test_state_hash_excludes_snapshot_step_and_provenance() -> None:
    torch = pytest.importorskip("torch")
    first = capture_scene_snapshot(
        _Env(torch, 1.0, 10),
        task_id="pick_and_place_box",
        event_id="first",
        episode_seed=1,
        scene_assets=("robot", "box"),
        simulator_revision="isaac-test",
    )
    second = capture_scene_snapshot(
        _Env(torch, 1.0, 0),
        task_id="pick_and_place_box",
        event_id="second",
        episode_seed=2,
        scene_assets=("robot", "box"),
        simulator_revision="isaac-test",
    )
    assert first["snapshot_sha256"] != second["snapshot_sha256"]
    assert snapshot_state_sha256(first) == snapshot_state_sha256(second)


def test_restore_is_read_back_before_policy_execution(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    source_env = _Env(torch, 2.5, 31)
    snapshot = capture_scene_snapshot(
        source_env,
        task_id="pick_and_place_box",
        event_id="initial",
        episode_seed=20260915,
        scene_assets=("robot", "box"),
        simulator_revision="isaac-test",
    )
    snapshot_path = tmp_path / "snapshot.json"
    write_snapshot_atomic(snapshot_path, snapshot)

    target_env = _Env(torch, -8.0, 0)
    report = restore_snapshot_for_trial(
        target_env,
        snapshot_path,
        expected_snapshot_sha256=snapshot["snapshot_sha256"],
        expected_task_id="pick_and_place_box",
        expected_event_id="initial",
        expected_simulator_revision="isaac-test",
        policy_rollout_seed=17,
        audit_path=tmp_path / "restore-audit.json",
    )
    assert report["restore_validated"] is True
    assert report["source_state_sha256"] == report["readback_state_sha256"]
    assert target_env.scene.write_count == 1
    assert target_env.sim.forward_count == 1
    torch.testing.assert_close(
        target_env.scene["robot"].data.root_state_w,
        source_env.scene["robot"].data.root_state_w,
    )


def test_restore_refuses_the_wrong_content_address(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    env = _Env(torch, 1.0, 0)
    snapshot = capture_scene_snapshot(
        env,
        task_id="pick_and_place_box",
        event_id="initial",
        episode_seed=20260915,
        scene_assets=("robot", "box"),
        simulator_revision="isaac-test",
    )
    snapshot_path = tmp_path / "snapshot.json"
    write_snapshot_atomic(snapshot_path, snapshot)
    with pytest.raises(ValueError, match="snapshot_sha256 differs"):
        restore_snapshot_for_trial(
            env,
            snapshot_path,
            expected_snapshot_sha256="0" * 64,
            expected_task_id="pick_and_place_box",
            expected_event_id="initial",
            expected_simulator_revision="isaac-test",
            policy_rollout_seed=17,
            audit_path=tmp_path / "restore-audit.json",
        )


def test_failure_start_trial_audit_binds_restore_to_result(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    scenario_id = "box-drop-and-body-push"
    simulator_revision = SUITE["reference_stack"]["isaac_lab_revision"]
    snapshot = capture_scene_snapshot(
        _Env(torch, 3.0, 25),
        task_id="pick_and_place_box",
        event_id=scenario_id,
        episode_seed=20260915,
        scene_assets=("robot", "box"),
        simulator_revision=simulator_revision,
    )
    snapshot_path = tmp_path / "snapshot.json"
    write_snapshot_atomic(snapshot_path, snapshot)
    restore = restore_snapshot_for_trial(
        _Env(torch, -2.0, 0),
        snapshot_path,
        expected_snapshot_sha256=snapshot["snapshot_sha256"],
        expected_task_id="pick_and_place_box",
        expected_event_id=scenario_id,
        expected_simulator_revision=simulator_revision,
        policy_rollout_seed=17,
        audit_path=tmp_path / "restore-audit.json",
    )
    summary = {
        "schema_version": 1,
        "suite_sha256": canonical_sha256(SUITE),
        "task_id": "pick_and_place_box",
        "scenario_id": scenario_id,
        "episode_seed": 17,
        "protocol": "failure_start",
        "failure_injected": False,
        "start_snapshot_sha256": snapshot["snapshot_sha256"],
        "restore_audit_sha256": restore["audit_sha256"],
        "restore_validated": True,
        "claim_boundary": "snapshot restore evidence; task success remains in the episode result",
    }
    (tmp_path / "trial-summary.json").write_text(json.dumps(summary), encoding="utf-8")
    result_path = tmp_path / "episode.json"
    result_path.write_text(
        json.dumps(
            {
                "episode_seed": 17,
                "success": True,
                "failure_reason": "success",
                "hrvla_recovery": summary,
            }
        ),
        encoding="utf-8",
    )
    report = audit_failure_start_trial(
        SUITE,
        scenario_id,
        tmp_path,
        result_path,
        expected_snapshot_sha256=snapshot["snapshot_sha256"],
    )
    assert report["status"] == "restore_and_behavioral_result_validated"
    assert report["success"] is True
