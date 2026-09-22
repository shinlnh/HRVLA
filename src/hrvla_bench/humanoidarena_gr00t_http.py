"""HTTP protocol adapter from HumanoidArena observations to Isaac-GR00T.

The simulator's ``SonicActionProvider`` sends one RGB image and the released
64-D reference-pose observation.  This module converts that flat contract to
GR00T's named modalities and converts the predicted named action groups back to
the exact 40-D semantic action consumed by SONIC.

The helpers deliberately have no torch or GR00T import so the complete wire
contract can be tested without loading a multi-gigabyte checkpoint.
"""

from __future__ import annotations

import base64
import json
import random
import threading
import traceback
from http.server import BaseHTTPRequestHandler
from typing import Any, Mapping
from urllib.parse import urlsplit

import numpy as np

from .humanoidarena_bridge import (
    ACTION_DIM,
    ACTION_FIELDS,
    LANGUAGE_KEY,
    STATE_DIM,
    STATE_FIELDS,
    VIDEO_KEY,
)


TASK_LANGUAGE_INSTRUCTIONS = {
    "HOI_double_desk": "Put the hammer from the right table into the basket on the left table.",
    "HOI_football": "Kick the soccer ball into the goal.",
    "HOI_pp_box": "Move the box from the table onto the shelf.",
    "HSI_vision_navi": "Avoid obstacles and move to the yellow marked area.",
    "HSI_open_door": "Open the door.",
    "HSI_sit_sofa": "Sit on the sofa.",
    "HOI_grap_cup": "Place the orange drink can on the table into the basket on the tea table.",
    "HSI_boxing": "Strike the green markers on the punching bag.",
}

_TASK_ALIASES = (
    ("doubledesk", "HOI_double_desk"),
    ("double_desk", "HOI_double_desk"),
    ("football", "HOI_football"),
    ("ppbox", "HOI_pp_box"),
    ("pp_box", "HOI_pp_box"),
    ("visionnavi", "HSI_vision_navi"),
    ("vision_navi", "HSI_vision_navi"),
    ("opendoor", "HSI_open_door"),
    ("open_door", "HSI_open_door"),
    ("sitsofa", "HSI_sit_sofa"),
    ("sit_sofa", "HSI_sit_sofa"),
    ("grapcup", "HOI_grap_cup"),
    ("grab_cup", "HOI_grap_cup"),
    ("boxing", "HSI_boxing"),
)


def _normalize_task(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def resolve_task_instruction(value: str | None) -> tuple[str | None, str | None]:
    """Resolve released task IDs/names while preserving explicit instructions."""

    if value is None:
        return None, None
    raw = str(value).strip()
    if not raw:
        return None, None
    if raw in TASK_LANGUAGE_INSTRUCTIONS:
        return raw, TASK_LANGUAGE_INSTRUCTIONS[raw]
    normalized = _normalize_task(raw)
    for task_name, instruction in TASK_LANGUAGE_INSTRUCTIONS.items():
        if normalized == _normalize_task(instruction):
            return task_name, instruction
    for alias, task_name in _TASK_ALIASES:
        if _normalize_task(alias) in normalized:
            return task_name, TASK_LANGUAGE_INSTRUCTIONS[task_name]
    return None, raw


def decode_front_image(payload: Mapping[str, Any]) -> np.ndarray:
    """Decode and strictly validate the image envelope used by the HA client."""

    shape = tuple(int(value) for value in payload["shape"])
    dtype = np.dtype(payload["dtype"])
    if len(shape) != 3 or shape[-1] != 3 or any(value <= 0 for value in shape):
        raise ValueError(f"front image must have shape (H,W,3), got {shape}")
    if dtype != np.dtype(np.uint8):
        raise ValueError(f"front image must be uint8, got {dtype}")
    raw = base64.b64decode(str(payload["data_b64"]), validate=True)
    expected_bytes = int(np.prod(shape)) * dtype.itemsize
    if len(raw) != expected_bytes:
        raise ValueError(f"front image has {len(raw)} bytes, expected {expected_bytes}")
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


def build_gr00t_observation(
    front_rgb: np.ndarray, observation_state: Any, instruction: str
) -> dict[str, Any]:
    """Build a strict B=1, T=1 GR00T observation without changing values."""

    image = np.asarray(front_rgb)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"front image must be uint8 (H,W,3), got {image.shape}/{image.dtype}")
    state = np.asarray(observation_state, dtype=np.float32)
    if state.shape != (STATE_DIM,):
        raise ValueError(f"observation state must have shape ({STATE_DIM},), got {state.shape}")
    if not np.isfinite(state).all():
        raise ValueError("observation state contains non-finite values")
    if not instruction.strip():
        raise ValueError("a non-empty task instruction is required")

    named_state = {
        field.name: state[field.start : field.end].reshape(1, 1, field.width).copy()
        for field in STATE_FIELDS
    }
    return {
        "video": {VIDEO_KEY: image.reshape(1, 1, *image.shape)},
        "state": named_state,
        "language": {LANGUAGE_KEY: [[instruction.strip()]]},
    }


def flatten_gr00t_action(action: Mapping[str, Any]) -> np.ndarray:
    """Flatten one GR00T named action batch to SONIC's ``(T, 40)`` contract."""

    expected_names = {field.name for field in ACTION_FIELDS}
    actual_names = set(action)
    if actual_names != expected_names:
        raise ValueError(
            "GR00T action group mismatch: "
            f"missing={sorted(expected_names - actual_names)}, "
            f"extra={sorted(actual_names - expected_names)}"
        )

    arrays: list[np.ndarray] = []
    horizon: int | None = None
    for field in ACTION_FIELDS:
        array = np.asarray(action[field.name], dtype=np.float32)
        if array.ndim != 3 or array.shape[0] != 1 or array.shape[2] != field.width:
            raise ValueError(
                f"GR00T action {field.name} must have shape (1,T,{field.width}), got {array.shape}"
            )
        if horizon is None:
            horizon = int(array.shape[1])
        elif array.shape[1] != horizon:
            raise ValueError("GR00T action groups have different horizons")
        arrays.append(array[0])
    if not horizon:
        raise ValueError("GR00T returned an empty action horizon")
    flattened = np.concatenate(arrays, axis=-1).astype(np.float32, copy=False)
    if flattened.shape != (horizon, ACTION_DIM):
        raise AssertionError(f"flattened action has unexpected shape {flattened.shape}")
    if not np.isfinite(flattened).all():
        raise ValueError("GR00T action contains non-finite values")
    return flattened


def validate_policy_contract(policy: Any) -> None:
    """Fail before serving if a checkpoint embeds a different HA modality contract."""

    config = policy.get_modality_config()
    expected = {
        "video": [VIDEO_KEY],
        "state": [field.name for field in STATE_FIELDS],
        "action": [field.name for field in ACTION_FIELDS],
        "language": [LANGUAGE_KEY],
    }
    for modality, keys in expected.items():
        actual = list(config[modality].modality_keys)
        if actual != keys:
            raise ValueError(f"checkpoint {modality} keys are {actual}, expected {keys}")
    if list(config["video"].delta_indices) != [0]:
        raise ValueError("checkpoint video horizon differs from [0]")
    if list(config["state"].delta_indices) != [0]:
        raise ValueError("checkpoint state horizon differs from [0]")
    if list(config["language"].delta_indices) != [0]:
        raise ValueError("checkpoint language horizon differs from [0]")
    if list(config["action"].delta_indices) != list(range(40)):
        raise ValueError("checkpoint action horizon differs from range(40)")


class HumanoidArenaGr00tState:
    """Serialize GPU inference and expose the released HA reset/infer semantics."""

    def __init__(self, policy: Any, default_instruction: str | None = None, instruction_router: Any = None):
        self.policy = policy
        self.instruction_router = instruction_router
        self.default_instruction = (default_instruction or "").strip() or None
        self.lock = threading.Lock()
        self.reset_count = 0
        self.infer_count = 0
        self.current_seed: int | None = None

    def reset(self, seed: int | None = None) -> int:
        with self.lock:
            if seed is not None:
                self.current_seed = int(seed) & 0xFFFFFFFFFFFFFFFF
                normalized_seed = self.current_seed & 0xFFFFFFFF
                random.seed(normalized_seed)
                np.random.seed(normalized_seed)
                try:
                    import torch

                    torch.manual_seed(normalized_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(normalized_seed)
                except ImportError:
                    pass
            self.policy.reset({"seed": self.current_seed})
            if self.instruction_router is not None:
                self.instruction_router.reset()
            self.reset_count += 1
            self.infer_count = 0
            return self.reset_count

    def infer_payload(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        observation_payload = payload["observation"]
        image = decode_front_image(observation_payload["images"]["front"])
        state = np.asarray(observation_payload["state"], dtype=np.float32)
        task_value = payload.get("task", payload.get("task_name"))
        task_name, instruction = resolve_task_instruction(task_value)
        instruction = instruction or self.default_instruction
        if instruction is None:
            raise ValueError("request has no task and server has no default instruction")
        with self.lock:
            skill_id = None
            if self.instruction_router is not None:
                instruction, skill_id = self.instruction_router.select(task_value, payload)
            observation = build_gr00t_observation(image, state, instruction)
            result = self.policy.get_action(observation)
            action = result[0] if isinstance(result, tuple) else result
            action_chunk = flatten_gr00t_action(action)
            self.infer_count += 1
            infer_count = self.infer_count
        return {
            "action": action_chunk[0].tolist(),
            "action_chunk": action_chunk.tolist(),
            "chunk_size": int(action_chunk.shape[0]),
            "task_name": task_name,
            "instruction": instruction,
            "selected_skill_id": skill_id,
            "infer_count": infer_count,
        }


def make_handler(state: HumanoidArenaGr00tState, max_body_bytes: int = 8 * 1024 * 1024):
    """Create a quiet standard-library handler compatible with HA's HTTP client."""

    class Handler(BaseHTTPRequestHandler):
        def _read_json(self) -> dict[str, Any]:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length < 0 or content_length > max_body_bytes:
                raise ValueError(f"request body size {content_length} exceeds limit")
            raw = self.rfile.read(content_length) if content_length else b"{}"
            value = json.loads(raw.decode("utf-8"))
            if not isinstance(value, dict):
                raise ValueError("JSON body must be an object")
            return value

        def _send_json(self, status: int, payload: Mapping[str, Any]) -> None:
            encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:  # noqa: N802
            if urlsplit(self.path).path != "/healthz":
                self._send_json(404, {"error": f"unknown path: {self.path}"})
                return
            self._send_json(
                200,
                {
                    "ok": True,
                    "reset_count": state.reset_count,
                    "infer_count": state.infer_count,
                },
            )

        def do_POST(self) -> None:  # noqa: N802
            path = urlsplit(self.path).path
            try:
                payload = self._read_json()
                if path == "/reset":
                    state.reset(payload.get("seed"))
                    self._send_json(200, {"ok": True})
                    return
                if path != "/infer":
                    self._send_json(404, {"error": f"unknown path: {self.path}"})
                    return
                self._send_json(200, state.infer_payload(payload))
            except (AssertionError, KeyError, TypeError, ValueError) as error:
                self._send_json(400, {"error": str(error), "type": type(error).__name__})
            except BrokenPipeError:
                return
            except Exception as error:  # pragma: no cover - production diagnostic path
                traceback.print_exc()
                self._send_json(500, {"error": str(error), "type": type(error).__name__})

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler
