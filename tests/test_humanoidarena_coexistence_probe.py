import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/probe_humanoidarena_gr00t_coexistence.py"
SPEC = importlib.util.spec_from_file_location("coexistence_probe", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def test_probe_job_forces_real_short_camera_episode(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-200"
    job = PROBE.build_probe_job(tmp_path / "evidence", checkpoint, steps=40, seed=17)
    assert job["max_steps"] == 40
    assert job["episode_seed"] == 17
    assert job["video_fps"] == 30
    assert job["post_termination_record_steps"] == 2
    assert job["result_json"].endswith("episode.json")
