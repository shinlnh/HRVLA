from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys


ROOT = Path(__file__).resolve().parents[1]


def _load_launcher():
    spec = importlib.util.spec_from_file_location(
        "train_gr00t_vla_adapter_test", ROOT / "scripts/train_gr00t_vla_adapter.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_final_root_save_is_skipped_only_after_complete_numbered_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    calls = []

    class FakeTrainer:
        def save_model(self, output_dir=None, *args, **kwargs):
            calls.append(output_dir)
            return "saved"

    gr00t = ModuleType("gr00t")
    experiment = ModuleType("gr00t.experiment")
    trainer_module = ModuleType("gr00t.experiment.trainer")
    trainer_module.Gr00tTrainer = FakeTrainer
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    monkeypatch.setitem(sys.modules, "gr00t", gr00t)
    monkeypatch.setitem(sys.modules, "gr00t.experiment", experiment)
    monkeypatch.setitem(sys.modules, "gr00t.experiment.trainer", trainer_module)

    launcher = _load_launcher()
    launcher._install_redundant_final_save_guard(300, 100)
    trainer = FakeTrainer()
    trainer.args = SimpleNamespace(output_dir=str(tmp_path))
    trainer.state = SimpleNamespace(global_step=300)

    assert trainer.save_model() == "saved"
    assert calls == [None]
    final = tmp_path / "checkpoint-300"
    final.mkdir()
    (final / "trainer_state.json").write_text('{"global_step": 300}\n')
    assert trainer.save_model() is None
    assert calls == [None]
    assert trainer.save_model(str(final)) == "saved"
    assert calls == [None, str(final)]
