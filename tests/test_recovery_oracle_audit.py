from pathlib import Path

import pytest

from scripts.audit_humanoidarena_recovery_oracle import _trial_files


def test_trial_files_requires_one_complete_immutable_pair(tmp_path: Path) -> None:
    scenario = tmp_path / "scenario"
    with pytest.raises(ValueError, match="expected exactly one"):
        _trial_files(scenario, 0)

    trial = scenario / "attempt-0001" / "trial-0000"
    trial.mkdir(parents=True)
    (trial / "oracle-record.json").write_text("{}")
    with pytest.raises(ValueError, match="episode JSON missing"):
        _trial_files(scenario, 0)

    episode = trial / "episode.json"
    episode.write_text("{}")
    assert _trial_files(scenario, 0) == (trial / "oracle-record.json", episode)

    duplicate = scenario / "attempt-0002" / "trial-0000"
    duplicate.mkdir(parents=True)
    (duplicate / "oracle-record.json").write_text("{}")
    with pytest.raises(ValueError, match="expected exactly one"):
        _trial_files(scenario, 0)
