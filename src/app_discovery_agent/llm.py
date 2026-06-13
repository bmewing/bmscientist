from __future__ import annotations

import json
import re
import time
from typing import TypeVar

from openai import OpenAI
from openai import APITimeoutError
from pydantic import BaseModel

from app_discovery_agent.config import AppConfig


ModelT = TypeVar("ModelT", bound=BaseModel)


class DeepSeekLLM:
    def __init__(self, config: AppConfig, model: str | None = None):
        base_url = config.deepseek_base_url.rstrip("/")
        self._client = OpenAI(
            api_key=config.deepseek_api_key,
            base_url=f"{base_url}/",
            timeout=config.request_timeout_seconds,
        )
        self._model = model or config.chat_model
        self._max_attempts = 3

    def _create_completion(self, **kwargs):
        last_error: Exception | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                return self._client.chat.completions.create(**kwargs)
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
            model=self._model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
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
            model=self._model,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content or "{}"
        payload = self._coerce_json(content)
        return response_model.model_validate(payload)

    @staticmethod
    def _coerce_json(content: str) -> dict:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, flags=re.DOTALL)
            if not match:
                raise
            return json.loads(match.group(0))
