from __future__ import annotations

import json
import re
from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel

from app_discovery_agent.config import AppConfig


ModelT = TypeVar("ModelT", bound=BaseModel)


class DeepSeekLLM:
    def __init__(self, config: AppConfig):
        base_url = config.deepseek_base_url.rstrip("/")
        self._client = OpenAI(
            api_key=config.deepseek_api_key,
            base_url=f"{base_url}/",
            timeout=config.request_timeout_seconds,
        )
        self._model = config.chat_model

    def complete_text(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
        response = self._client.chat.completions.create(
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
        response = self._client.chat.completions.create(
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

