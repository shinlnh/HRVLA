import copy
import json
import unittest
from pathlib import Path

from hrvla_bench.methods import validate_method_registry


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "benchmark/methods/registry.json"
ARTIFACT_LOCK = ROOT / "config/benchmark-artifacts.lock.json"


class MethodRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
        self.artifact_lock = json.loads(ARTIFACT_LOCK.read_text(encoding="utf-8"))

    def test_registry_and_artifact_publications_are_consistent(self) -> None:
        validate_method_registry(self.registry, self.artifact_lock, ROOT)

    def test_suffix_taxonomy_cannot_be_relabelled(self) -> None:
        registry = copy.deepcopy(self.registry)
        method = next(item for item in registry["methods"] if item["id"] == "gr00t_st")
        method["features"]["recovery"] = True
        with self.assertRaisesRegex(ValueError, "taxonomy"):
            validate_method_registry(registry, self.artifact_lock, ROOT)

    def test_retrained_checkpoint_must_match_published_lock(self) -> None:
        registry = copy.deepcopy(self.registry)
        method = next(item for item in registry["methods"] if item["id"] == "gr00t_str_rt")
        method["checkpoint"]["revision"] = "0" * 40
        with self.assertRaisesRegex(ValueError, "does not match publication"):
            validate_method_registry(registry, self.artifact_lock, ROOT)


if __name__ == "__main__":
    unittest.main()
