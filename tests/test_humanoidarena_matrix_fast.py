from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_humanoidarena_baseline_matrix_fast.py"
SPEC = importlib.util.spec_from_file_location("humanoidarena_matrix_fast", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
FAST = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FAST)


def test_pending_repeat_ids_reuses_atomic_episode_evidence(tmp_path: Path) -> None:
    episodes = tmp_path / "episodes"
    episodes.mkdir()
    for repeat_idx, reason in ((0, "success"), (2, "timeout"), (3, "interrupted")):
        (episodes / f"episode-{repeat_idx}.json").write_text(
            json.dumps(
                {
                    "seed": 1,
                    "repeat_idx": repeat_idx,
                    "success": reason == "success",
                    "failure_reason": reason,
                }
            )
            + "\n",
            encoding="utf-8",
        )
    assert FAST.pending_repeat_ids(tmp_path, seed=1, repeats=4) == [1, 3]


def test_episode_seed_matches_locked_upstream_formula() -> None:
    assert FAST._derive_episode_seed("task", 0, 0) == 659473100
    assert FAST._derive_episode_seed("task", 2, 19) == 1207044323


def test_sim_command_preserves_locked_runtime_contract(tmp_path: Path) -> None:
    command = FAST._sim_command(
        task_name="boxing",
        mode="base_test",
        batch_path=tmp_path / "batch.json",
        port=18443,
        record_video_every_n=10,
        step_log_every_n=250,
    )
    assert command[command.index("--device") + 1] == "cuda:0"
    assert command[command.index("--sonic_vla_root_rot6d_layout") + 1] == "row"
    assert command[command.index("--record_video_every_n") + 1] == "10"
    assert "--episode_batch_json" in command


def test_default_fast_matrix_keeps_full_paper_sample_size() -> None:
    assert len(FAST.TASKS) * len(FAST.MODES) * len(FAST.SEEDS) * FAST.REPEATS == 1680
    assert FAST.DEFAULT_POLICY_THREADS == min(24, len(FAST.os.sched_getaffinity(0)))


def test_cpu_server_keeps_upstream_cuda_discovery_contract(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    env = FAST._server_env(cpu_threads=32, interop_threads=2, compile_threads=32)
    assert "CUDA_VISIBLE_DEVICES" not in env
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert env["OMP_NUM_THREADS"] == "32"


def test_cuda_int8_server_uses_gpu_and_explicit_backend() -> None:
    command = FAST._server_command(
        Path("/tmp/model"), 18443, FAST.CUDA_INT8_BACKEND
    )
    env = FAST._server_env(24, 2, 32, FAST.CUDA_INT8_BACKEND)
    assert command[command.index("--device") + 1] == "cuda:0"
    assert env["HRVLA_PI05_BACKEND"] == FAST.CUDA_INT8_BACKEND


def test_progress_binds_backend_provenance(tmp_path: Path) -> None:
    progress = FAST._matrix_progress(
        tmp_path,
        tasks=["boxing"],
        modes=["base_test"],
        seeds=[0],
        repeats=1,
        policy_backend=FAST.CUDA_INT8_BACKEND,
    )
    assert progress["policy_backend"] == FAST.CUDA_INT8_BACKEND
    assert progress["policy_device"] == "cuda:0"


def test_terminate_group_is_noop_after_process_exit() -> None:
    class ExitedProcess:
        def poll(self):
            return 0

    FAST._terminate_group(ExitedProcess())


def test_resume_reconciles_complete_cell_without_summary(tmp_path: Path) -> None:
    output_root = tmp_path / "matrix"
    cell_dir = output_root / "vision" / "boxing" / "seed-1"
    episodes = cell_dir / "episodes"
    episodes.mkdir(parents=True)
    for repeat_idx in range(3):
        (episodes / f"episode-{repeat_idx}.json").write_text(
            json.dumps(
                {
                    "seed": 1,
                    "repeat_idx": repeat_idx,
                    "success": repeat_idx == 0,
                    "failure_reason": "success" if repeat_idx == 0 else "timeout",
                    "returncode": 0,
                }
            )
            + "\n",
            encoding="utf-8",
        )

    reconciled = FAST._reconcile_complete_cells(
        output_root,
        task_name="boxing",
        modes=["vision"],
        seeds=[1],
        repeats=3,
        model_path=tmp_path / "checkpoint",
        log_root=tmp_path / "logs" / "boxing",
    )

    assert reconciled == 1
    assert len((cell_dir / "summary.jsonl").read_text().splitlines()) == 3
    summary = json.loads((cell_dir / "summary.json").read_text())
    assert summary["episodes"] == 3
    assert summary["successes"] == 1
    assert (
        FAST._reconcile_complete_cells(
            output_root,
            task_name="boxing",
            modes=["vision"],
            seeds=[1],
            repeats=3,
            model_path=tmp_path / "checkpoint",
            log_root=tmp_path / "logs" / "boxing",
        )
        == 0
    )
