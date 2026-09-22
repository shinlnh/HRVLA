from __future__ import annotations

import base64
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from hrvla_bench.humanoidarena_bridge import ACTION_FIELDS, STATE_FIELDS
from hrvla_bench.humanoidarena_gr00t_http import (
    HumanoidArenaGr00tState,
    build_gr00t_observation,
    flatten_gr00t_action,
    make_handler,
    resolve_task_instruction,
)


class FakePolicy:
    def __init__(self) -> None:
        self.observation = None
        self.reset_options = []

    def reset(self, options=None):
        self.reset_options.append(options)
        return {}

    def get_action(self, observation):
        self.observation = observation
        actions = {
            field.name: np.full((1, 40, field.width), field.start, dtype=np.float32)
            for field in ACTION_FIELDS
        }
        return actions, {"ignored": True}


def _payload(task: str = "Isaac-Move-Boxing-Bag-G129-Dex3-Wholebody") -> dict:
    image = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)
    return {
        "observation": {
            "images": {
                "front": {
                    "shape": list(image.shape),
                    "dtype": str(image.dtype),
                    "data_b64": base64.b64encode(image.tobytes()).decode("ascii"),
                }
            },
            "state": np.arange(64, dtype=np.float32).tolist(),
        },
        "robot_type": "g129",
        "task": task,
        "return_chunk": True,
    }


def test_task_router_accepts_isaac_id_and_preserves_explicit_instruction() -> None:
    name, instruction = resolve_task_instruction("Isaac-Move-Boxing-Bag-G129-Dex3-Wholebody")
    assert name == "HSI_boxing"
    assert instruction == "Strike the green markers on the punching bag."
    assert resolve_task_instruction("Do the custom task.") == (None, "Do the custom task.")


def test_observation_bridge_has_exact_named_shapes_and_values() -> None:
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    state = np.arange(64, dtype=np.float32)
    observation = build_gr00t_observation(image, state, "test task")
    assert observation["video"]["front"].shape == (1, 1, 12, 16, 3)
    for field in STATE_FIELDS:
        values = observation["state"][field.name]
        assert values.shape == (1, 1, field.width)
        np.testing.assert_array_equal(values[0, 0], state[field.start : field.end])
    assert observation["language"]["annotation.human.task_description"] == [["test task"]]


def test_action_bridge_reconstructs_fixed_field_order() -> None:
    action = {
        field.name: np.tile(
            np.arange(field.start, field.end, dtype=np.float32), (1, 40, 1)
        )
        for field in ACTION_FIELDS
    }
    flattened = flatten_gr00t_action(action)
    assert flattened.shape == (40, 40)
    np.testing.assert_array_equal(flattened[0], np.arange(40, dtype=np.float32))
    broken = dict(action)
    broken["hand_binary"] = np.zeros((1, 39, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="different horizons"):
        flatten_gr00t_action(broken)


def test_http_reset_and_infer_match_humanoidarena_client_protocol() -> None:
    policy = FakePolicy()
    state = HumanoidArenaGr00tState(policy)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"

    def post(path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            base_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=3) as response:
            return json.loads(response.read())

    try:
        assert post("/reset", {"seed": 17}) == {"ok": True}
        response = post("/infer", _payload())
        assert response["task_name"] == "HSI_boxing"
        assert response["chunk_size"] == 40
        assert len(response["action_chunk"]) == 40
        assert len(response["action_chunk"][0]) == 40
        assert policy.observation["state"]["joint_pos"].dtype == np.float32

        malformed = _payload()
        malformed["observation"]["state"] = [0.0] * 63
        with pytest.raises(urllib.error.HTTPError) as caught:
            post("/infer", malformed)
        assert caught.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
