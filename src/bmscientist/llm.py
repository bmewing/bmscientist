from __future__ import annotations

import json
import logging
import re
import time
from typing import TypeVar

from openai import OpenAI
from openai import APITimeoutError
from pydantic import BaseModel, ValidationError

from bmscientist.config import AppConfig, DeepSeekRequestProfile
from bmscientist.cost_tracking import CostTracker


ModelT = TypeVar("ModelT", bound=BaseModel)
LOGGER = logging.getLogger(__name__)


class DeepSeekLLM:
    def __init__(
        self,
        config: AppConfig,
        model: str | None = None,
        request_profile: DeepSeekRequestProfile | None = None,
        cost_tracker: CostTracker | None = None,
        client_name: str | None = None,
    ):
        base_url = config.deepseek_base_url.rstrip("/")
        profile = request_profile or config.chat_profile
        self._request_profile = config._resolved_profile(profile, model)
        self._cost_tracker = cost_tracker
        self._client_name = client_name
        self._client = OpenAI(
            api_key=config.deepseek_api_key,
            base_url=f"{base_url}/",
            timeout=self._request_profile.timeout_seconds or config.request_timeout_seconds,
        )
        self._model = self._request_profile.model or config.chat_model
        self._max_attempts = 3
        self._max_validation_repairs = 2

    def _create_completion(self, **kwargs):
        last_error: Exception | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                response = self._client.chat.completions.create(**kwargs)
                if self._cost_tracker is not None:
                    self._cost_tracker.record_deepseek_response(
                        response=response,
                        model=self._model,
                        client_name=self._client_name,
                    )
                return response
            except APITimeoutError as exc:
                last_error = exc
                if attempt >= self._max_attempts:
                    raise
                time.sleep(min(attempt * 2, 6))
        if last_error:
            raise last_error
        raise RuntimeError("Completion request failed without an explicit error.")

    def complete_text(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
        response = self._create_completion(
            **self._completion_kwargs(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
            )
        )
        return response.choices[0].message.content or ""

    def complete_json(
        self,
        response_model: type[ModelT],
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
    ) -> ModelT:
        response = self._create_completion(
            **self._completion_kwargs(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                response_format={"type": "json_object"},
            )
        )
        content = response.choices[0].message.content or "{}"
        try:
            payload = self._coerce_json(content)
        except json.JSONDecodeError as exc:
            LOGGER.warning(
                "Malformed JSON from %s; attempting repair. Error: %s",
                response_model.__name__,
                exc,
            )
            repaired_content = self._repair_json_content(response_model, content)
            payload = self._coerce_json(repaired_content)
        return self._validate_json_payload(
            response_model=response_model,
            payload=payload,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

    def _repair_json_content(self, response_model: type[ModelT], content: str) -> str:
        schema = json.dumps(response_model.model_json_schema(), indent=2)
        response = self._create_completion(
            **self._completion_kwargs(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You repair malformed JSON. "
                            "Return strict JSON only, preserve the original meaning, and do not add commentary."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Repair the malformed JSON-like content below so it matches the target schema.\n\n"
                            f"Target schema:\n{schema}\n\n"
                            f"Malformed content:\n{content}"
                        ),
                    },
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
        )
        return response.choices[0].message.content or "{}"

    def _validate_json_payload(
        self,
        *,
        response_model: type[ModelT],
        payload: dict[str, object],
        system_prompt: str,
        user_prompt: str,
    ) -> ModelT:
        current_payload = payload
        last_error: ValidationError | None = None
        max_validation_repairs = getattr(self, "_max_validation_repairs", 2)
        for attempt in range(max_validation_repairs + 1):
            try:
                return response_model.model_validate(current_payload)
            except ValidationError as exc:
                last_error = exc
                if attempt >= max_validation_repairs:
                    raise
                LOGGER.warning(
                    "Pydantic validation failed for %s; attempting schema repair pass %s/%s. Error: %s",
                    response_model.__name__,
                    attempt + 1,
                    max_validation_repairs,
                    exc,
                )
                repaired_content = self._repair_validation_errors(
                    response_model=response_model,
                    payload=current_payload,
                    validation_error=exc,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
                current_payload = self._coerce_json(repaired_content)
        if last_error:
            raise last_error
        raise RuntimeError("Validation failed without an explicit error.")

    def _repair_validation_errors(
        self,
        *,
        response_model: type[ModelT],
        payload: dict[str, object],
        validation_error: ValidationError,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        schema = json.dumps(response_model.model_json_schema(), indent=2)
        error_payload = json.dumps(validation_error.errors(), indent=2)
        original_payload = json.dumps(payload, indent=2, ensure_ascii=False)
        response = self._create_completion(
            **self._completion_kwargs(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You repair JSON so it satisfies a Pydantic schema for the same task context. "
                            "Return strict JSON only. Preserve the original meaning whenever possible. "
                            "Do not add commentary. Do not invent evidence, citations, tool outputs, or unsupported facts. "
                            "If a field cannot be justified from the original payload and task context, prefer null, empty collections, "
                            "or conservative normalization over fabrication."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "The JSON below parsed successfully, but failed Pydantic validation.\n\n"
                            f"Original system prompt:\n{system_prompt}\n\n"
                            f"Original user prompt:\n{user_prompt}\n\n"
                            f"Target schema:\n{schema}\n\n"
                            f"Pydantic validation errors:\n{error_payload}\n\n"
                            f"Original JSON payload:\n{original_payload}\n\n"
                            "Repair the JSON so it satisfies the schema while preserving intent."
                        ),
                    },
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
        )
        return response.choices[0].message.content or "{}"

    def _completion_kwargs(self, *, messages: list[dict[str, str]], temperature: float, **kwargs) -> dict[str, object]:
        request: dict[str, object] = {
            "model": self._model,
            "messages": messages,
            **kwargs,
        }
        profile = getattr(self, "_request_profile", None)
        thinking = profile.thinking if profile is not None else None
        if thinking is not None:
            request["extra_body"] = {"thinking": {"type": "enabled" if thinking.enabled else "disabled"}}
            if thinking.enabled and thinking.effort:
                request["reasoning_effort"] = thinking.effort
            if not thinking.enabled:
                request["temperature"] = temperature
        else:
            request["temperature"] = temperature
        return request

    @staticmethod
    def _coerce_json(content: str) -> dict:
        cleaned = content.strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        candidates = DeepSeekLLM._json_candidates(cleaned)
        last_error: json.JSONDecodeError | None = None
        for candidate in candidates:
            for variant in DeepSeekLLM._json_variants(candidate):
                try:
                    return json.loads(variant)
                except json.JSONDecodeError as exc:
                    last_error = exc
        if last_error:
            raise last_error
        raise json.JSONDecodeError("No JSON object found", content, 0)

    @staticmethod
    def _json_candidates(content: str) -> list[str]:
        candidates: list[str] = []
        fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", content, flags=re.DOTALL | re.IGNORECASE)
        candidates.extend(fenced)
        balanced = DeepSeekLLM._balanced_json_object(content)
        if balanced:
            candidates.append(balanced)
        greedy_match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if greedy_match:
            candidates.append(greedy_match.group(0))
        unique: list[str] = []
        for candidate in candidates:
            stripped = candidate.strip()
            if stripped and stripped not in unique:
                unique.append(stripped)
        return unique

    @staticmethod
    def _balanced_json_object(content: str) -> str | None:
        start = content.find("{")
        if start < 0:
            return None
        depth = 0
        in_string = False
        escaped = False
        for index, character in enumerate(content[start:], start=start):
            if escaped:
                escaped = False
                continue
            if character == "\\":
                escaped = True
                continue
            if character == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    return content[start : index + 1]
        return None

    @staticmethod
    def _json_variants(candidate: str) -> list[str]:
        stripped = candidate.strip()
        no_trailing_commas = re.sub(r",\s*([}\]])", r"\1", stripped)
        python_literals = (
            no_trailing_commas.replace(": None", ": null")
            .replace(": True", ": true")
            .replace(": False", ": false")
        )
        escaped_controls = DeepSeekLLM._escape_control_chars_in_strings(python_literals)
        variants = [stripped, no_trailing_commas, python_literals, escaped_controls]
        unique: list[str] = []
        for variant in variants:
            if variant and variant not in unique:
                unique.append(variant)
        return unique

    @staticmethod
    def _escape_control_chars_in_strings(candidate: str) -> str:
        escaped_parts: list[str] = []
        in_string = False
        escaped = False
        for character in candidate:
            if escaped:
                escaped_parts.append(character)
                escaped = False
                continue
            if character == "\\":
                escaped_parts.append(character)
                escaped = True
                continue
            if character == '"':
                escaped_parts.append(character)
                in_string = not in_string
                continue
            if in_string and ord(character) < 32:
                replacements = {
                    "\b": "\\b",
                    "\f": "\\f",
                    "\n": "\\n",
                    "\r": "\\r",
                    "\t": "\\t",
                }
                escaped_parts.append(replacements.get(character, f"\\u{ord(character):04x}"))
                continue
            escaped_parts.append(character)
        return "".join(escaped_parts)
