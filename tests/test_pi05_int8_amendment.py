import pytest

from scripts.compile_humanoidarena_pi05_int8_amendment import (
    _assert_corrected_route,
    _selected_root,
)


def test_corrected_task_only_uses_amended_root(tmp_path):
    original = tmp_path / "original"
    corrected = tmp_path / "corrected"
    assert _selected_root("pp_box", original, corrected) == corrected
    assert _selected_root("boxing", original, corrected) == original


def test_route_audit_requires_exact_instruction():
    good = "[lerobot_vla_server] first_infer peer=('127.0.0.1', 1) task_name=HOI_pp_box task='Move the box from the table onto the shelf.' state_shape=(64,)"
    assert _assert_corrected_route(good) == 1
    with pytest.raises(ValueError):
        _assert_corrected_route(good + "\n[lerobot_vla_server] first_infer peer=x task_name=None task='Isaac-Move-PickPlace-Box-G129-Dex3-Wholedoby'")
