from __future__ import annotations

import json
from pathlib import Path
import subprocess

from hrvla_bench.humanoidarena import admission_report


ROOT = Path(__file__).resolve().parents[1]
LOCK = json.loads((ROOT / "config/humanoidarena-admission.lock.json").read_text())


def _materialize_release(root: Path) -> tuple[Path, Path, Path, Path, Path, Path, str]:
    source = root / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(
        ["git", "-C", str(source), "config", "user.email", "test@example.com"], check=True
    )
    subprocess.run(["git", "-C", str(source), "config", "user.name", "Test"], check=True)
    for relative in LOCK["required_source_paths"]:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    for task in LOCK["tasks"]:
        path = source / f"isaaclab_twist2_g1/batch_test_scripts/task/batch_test_{task}.sh"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    for mode in LOCK["evaluation_modes"]:
        (source / f"isaaclab_twist2_g1/tasks/common_test_config/{mode}").mkdir(parents=True)
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(["git", "-C", str(source), "commit", "-qm", "fixture"], check=True)
    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    models = root / "models"
    task_dirs = [
        "HSI_boxing",
        "HOI_double_desk",
        "HOI_football",
        "HSI_open_door",
        "HOI_pp_box",
        "HSI_sit_sofa",
        "HSI_vision_navi",
    ]
    for task in task_dirs:
        path = models / "pi" / task / "fixture/pretrained_model/model.safetensors"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    dataset = root / "dataset"
    for task in task_dirs:
        info = dataset / task / "sonic_refpose_v3_1/meta/info.json"
        info.parent.mkdir(parents=True, exist_ok=True)
        info.write_text("{}\n")
    assets = root / "assets"
    for directory in LOCK["resources"]["assets"]["required_directories"]:
        (assets / directory).mkdir(parents=True)
    sonic = root / "sonic"
    sonic.mkdir()
    for filename in LOCK["resources"]["sonic_policy"]["required_files"]:
        (sonic / filename).touch()
    oracle = root / "oracle.json"
    oracle.write_text('{"trials": 20, "successes": 20}\n')
    return source, models, dataset, assets, sonic, oracle, revision


def test_admission_requires_every_gate(tmp_path: Path) -> None:
    source, models, dataset, assets, sonic, oracle, revision = _materialize_release(tmp_path)
    lock = json.loads(json.dumps(LOCK))
    lock["source"]["revision"] = revision
    lock["upstream_self_test"]["status_at_lock"] = "passing"
    report = admission_report(
        lock,
        tmp_path,
        source_root=source,
        model_root=models,
        dataset_root=dataset,
        asset_root=assets,
        sonic_policy_root=sonic,
        oracle_evidence=oracle,
        isaac_sim_version="5.0.0",
        isaac_lab_ref="release/2.2.0",
    )
    assert report["admitted"] is True
    assert report["blockers"] == []


def test_admission_reports_runtime_artifact_and_oracle_blockers(tmp_path: Path) -> None:
    report = admission_report(LOCK, tmp_path, source_root=tmp_path / "missing")
    assert report["admitted"] is False
    assert "source_revision" in report["blockers"]
    assert "upstream_self_test" in report["blockers"]
    assert "isaac_sim_version" in report["blockers"]
    assert "released_dataset:boxing" in report["blockers"]
    assert "oracle_admission" in report["blockers"]
