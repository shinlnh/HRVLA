import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/render_humanoidarena_video_evidence.py"
SPEC = importlib.util.spec_from_file_location("video_evidence", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
VIDEO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VIDEO)


def test_representative_selection_prefers_success_then_is_deterministic(
    tmp_path: Path, monkeypatch
) -> None:
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    monkeypatch.setattr(VIDEO, "ROOT", tmp_path)
    rows = [
        {"task": "pick", "success": False, "video_path": str(first)},
        {"task": "pick", "success": True, "video_path": str(second)},
    ]
    selected = VIDEO.choose_representatives(rows, ("task",))
    assert len(selected) == 1
    assert selected[0]["success"] is True
    assert selected[0]["_resolved_video"] == str(second)
