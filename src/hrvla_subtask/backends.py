"""Proposal backends for high-level subtask generation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import re
from typing import Protocol

from .model import Candidate, ExecutionMemory, TaskSpec


class ProposalBackend(Protocol):
    """The planner only requires subtask proposals, not low-level actions."""

    @property
    def calls(self) -> int: ...

    @property
    def generated_tokens(self) -> int: ...

    def propose(
        self,
        task: TaskSpec,
        state: frozenset[str],
        memory: ExecutionMemory,
        count: int,
        seed: int,
    ) -> list[Candidate]: ...


@dataclass
class HeuristicProposalBackend:
    """Seeded noisy proposal model used to isolate planning-algorithm effects."""

    error_rate: float = 0.28
    _calls: int = 0
    _generated_tokens: int = 0

    @property
    def calls(self) -> int:
        return self._calls

    @property
    def generated_tokens(self) -> int:
        return self._generated_tokens

    def propose(
        self,
        task: TaskSpec,
        state: frozenset[str],
        memory: ExecutionMemory,
        count: int,
        seed: int,
    ) -> list[Candidate]:
        self._calls += 1
        rng = random.Random(seed)
        correct = task.canonical_next(state)
        pool = [skill for skill in task.skills if skill.milestone not in state]
        if not pool:
            return [Candidate("done", 0.99, "goal complete")]

        output: list[Candidate] = []
        for index in range(max(1, count)):
            choose_wrong = correct is None or rng.random() < self.error_rate
            if index > 0 and correct is not None and rng.random() < 0.55:
                choose_wrong = False
            if choose_wrong:
                skill = rng.choice(pool)
                confidence = rng.uniform(0.32, 0.78)
            else:
                skill = correct
                confidence = rng.uniform(0.64, 0.97)
            output.append(
                Candidate(
                    skill_id=skill.skill_id,
                    confidence=confidence,
                    rationale="seeded proposal model",
                )
            )
        self._generated_tokens += 12 * len(output)
        return output


class CosmosReasonBackend:
    """Local NVIDIA Cosmos-Reason2 proposal backend with token-confidence routing."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_new_tokens: int = 192,
    ) -> None:
        try:
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:  # pragma: no cover - depends on optional GPU environment
            raise RuntimeError("install the 'gpu' extra to use CosmosReasonBackend") from exc

        self._torch = torch
        self._device = device
        self._max_new_tokens = max_new_tokens
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        model_dtype = getattr(torch, dtype)
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(model_path),
            dtype=model_dtype,
            device_map=device,
            attn_implementation="sdpa",
            local_files_only=Path(model_path).exists(),
        ).eval()
        self._processor = AutoProcessor.from_pretrained(
            str(model_path), local_files_only=Path(model_path).exists()
        )
        self._calls = 0
        self._generated_tokens = 0
        self.peak_vram_bytes = 0

    @property
    def calls(self) -> int:
        return self._calls

    @property
    def generated_tokens(self) -> int:
        return self._generated_tokens

    @staticmethod
    def _extract_json(text: str) -> dict[str, object] | None:
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        candidates = [fenced.group(1)] if fenced else []
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            candidates.append(text[start : end + 1])
        for candidate in candidates:
            try:
                value = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        return None

    def _prompt(
        self, task: TaskSpec, state: frozenset[str], memory: ExecutionMemory, count: int
    ) -> str:
        skills = [
            {
                "id": skill.skill_id,
                "instruction": skill.instruction,
                "requires": sorted(skill.requires),
                "adds": sorted(skill.adds),
            }
            for skill in task.skills
        ]
        executable = [
            {
                "id": skill.skill_id,
                "instruction": skill.instruction,
                "priority": skill.priority,
            }
            for skill in task.skills
            if skill.applicable(state) and skill.useful(state)
        ]
        return (
            "Select the immediate next executable robot subtask. The skill_id MUST be one of "
            "EXECUTABLE_NOW; other skills are context for future steps only. Prefer prerequisite "
            "order and the lowest priority number when several choices make equal goal progress. "
            "Respect observed state and completed milestones. Do not invent skills. "
            f"Return JSON only as {{\"skill_id\":\"id\",\"confidence\":0..1,"
            "\"reason\":\"short\"}}.\n"
            f"GOAL: {task.goal_instruction}\n"
            f"OBSERVED_STATE: {json.dumps(sorted(state))}\n"
            f"MEMORY: {json.dumps(memory.to_dict(), sort_keys=True)}\n"
            f"EXECUTABLE_NOW: {json.dumps(executable, sort_keys=True)}\n"
            f"AVAILABLE_SKILLS: {json.dumps(skills, sort_keys=True)}\n"
            f"Generate candidate {count} of an independent proposal set."
        )

    def propose(
        self,
        task: TaskSpec,
        state: frozenset[str],
        memory: ExecutionMemory,
        count: int,
        seed: int,
    ) -> list[Candidate]:
        torch = self._torch
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are the high-level planner of a humanoid robot. "
                            "Choose one grounded subtask and emit strict JSON."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": self._prompt(task, state, memory, count)}],
            },
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(text=[text], return_tensors="pt").to(self._device)
        do_sample = count > 1
        kwargs: dict[str, object] = {
            "max_new_tokens": self._max_new_tokens,
            "do_sample": do_sample,
            "num_return_sequences": max(1, count),
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        if do_sample:
            kwargs.update({"temperature": 0.65, "top_p": 0.9})
        cuda_rng_devices = (
            [torch.cuda.current_device()]
            if torch.cuda.is_available() and self._device.startswith("cuda")
            else []
        )
        with torch.random.fork_rng(devices=cuda_rng_devices):
            torch.manual_seed(seed)
            with torch.inference_mode():
                generated = self._model.generate(**inputs, **kwargs)
        prompt_length = inputs.input_ids.shape[1]
        transition = self._model.compute_transition_scores(
            generated.sequences, generated.scores, normalize_logits=True
        )
        decoded = self._processor.batch_decode(
            generated.sequences[:, prompt_length:], skip_special_tokens=True
        )
        self._calls += 1
        if torch.cuda.is_available():
            self.peak_vram_bytes = max(self.peak_vram_bytes, torch.cuda.max_memory_allocated())

        known = task.skill_map
        output: list[Candidate] = []
        for index, raw_text in enumerate(decoded):
            payload = self._extract_json(raw_text) or {}
            skill_id = str(payload.get("skill_id", "")).strip()
            if not skill_id:
                skill_id = next((item for item in known if item in raw_text), "unknown")
            row_scores = transition[index]
            valid_scores = row_scores[torch.isfinite(row_scores) & row_scores.ne(0)]
            self._generated_tokens += int(valid_scores.numel())
            token_confidence = (
                float(torch.exp(valid_scores.mean()).item()) if valid_scores.numel() else 0.0
            )
            stated = payload.get("confidence", token_confidence)
            try:
                stated_confidence = float(stated)
            except (TypeError, ValueError):
                stated_confidence = token_confidence
            confidence = math.sqrt(max(0.0, stated_confidence) * token_confidence)
            output.append(
                Candidate(
                    skill_id=skill_id,
                    confidence=confidence,
                    rationale=str(payload.get("reason", "")),
                    raw_text=raw_text,
                ).normalized()
            )
        return output
