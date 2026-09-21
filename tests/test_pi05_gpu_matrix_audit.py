from scripts.audit_humanoidarena_pi05_gpu_matrix import _paired_sensitivity


def test_matched_cpu_gpu_sensitivity_counts_both_directions():
    base = {
        "episode_seed": 11,
        "episode_object_seed": 23,
        "task": "task",
        "max_steps": 100,
        "model_path": "model",
    }
    key0 = ("boxing", "base_test", 0, 0)
    key1 = ("boxing", "base_test", 0, 1)
    cpu = {key0: {**base, "success": True}, key1: {**base, "success": False}}
    gpu = {key0: {**base, "success": False}, key1: {**base, "success": True}}
    report = _paired_sensitivity(cpu, gpu)["overall"]
    assert report == {
        "pairs": 2,
        "cpu_successes": 1,
        "int8_successes": 1,
        "cpu_only": 1,
        "int8_only": 1,
        "paired_risk_difference_int8_minus_cpu": 0.0,
    }
