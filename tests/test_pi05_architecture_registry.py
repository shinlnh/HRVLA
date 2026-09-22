"""Keep the PI0.5 base architecture distinct from unimplemented variants."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pi05_sonic_base_manifest_is_honest() -> None:
    manifest = json.loads(
        (ROOT / "benchmark/methods/pi05_sonic.json").read_text(encoding="utf-8")
    )
    assert manifest["id"] == "pi05_sonic"
    assert manifest["features"] == {
        "subtask": False,
        "recovery": False,
        "retrained": False,
    }
    for key in (
        "server",
        "cuda_backend_implementation",
        "task_route_implementation",
    ):
        assert (ROOT / manifest["runtime"][key]).is_file()
    lock = json.loads(
        (ROOT / manifest["checkpoint_source"]["lock"]).read_text(encoding="utf-8")
    )
    assert manifest["checkpoint_source"]["repository"] == lock["resources"][
        "models"
    ]["repository"]
    assert manifest["evidence"]["simple"] is None
