from __future__ import annotations

import base64
import binascii
import json
import os
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _normalize_reasoning_effort(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"low", "medium", "med", "normal", "high"}:
        return "high"
    if text in {"xhigh", "very_high", "very-high", "max", "maximum"}:
        return "max"
    if text in {"none", "disabled", "off"}:
        return None
    return "high"


def _coerce_optional_secret_bytes(value: Any) -> bytes | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value or None
    text = str(value).strip()
    if not text:
        return None
    try:
        decoded_hex = bytes.fromhex(text)
    except ValueError:
        decoded_hex = None
    if decoded_hex:
        return decoded_hex
    try:
        decoded_b64 = base64.b64decode(text, validate=True)
    except (ValueError, binascii.Error):
        decoded_b64 = None
    if decoded_b64:
        return decoded_b64
    return text.encode("utf-8")


class DeepSeekThinkingConfig(BaseModel):
    enabled: bool = True
    effort: Literal["high", "max"] | None = None

    @field_validator("enabled", mode="before")
    @classmethod
    def normalize_enabled(cls, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        text = str(value or "").strip().lower()
        if text in {"", "true", "1", "yes", "on", "enabled", "enable"}:
            return True
        if text in {"false", "0", "no", "off", "disabled", "disable"}:
            return False
        return True

    @field_validator("effort", mode="before")
    @classmethod
    def normalize_effort(cls, value: Any) -> str | None:
        return _normalize_reasoning_effort(value)


class DeepSeekRequestProfile(BaseModel):
    model: str | None = None
    thinking: DeepSeekThinkingConfig | None = None
    timeout_seconds: int | None = Field(default=None, ge=5, le=600)

    @model_validator(mode="before")
    @classmethod
    def normalize_profile_aliases(cls, value: Any) -> Any:
        if value is None:
            return {}
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return {}
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"model": text}
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        if "thinking" not in payload:
            toggle = payload.get("thinking_enabled")
            effort = payload.get("thinking_effort") or payload.get("reasoning_effort")
            if toggle is not None or effort is not None:
                payload["thinking"] = {
                    "enabled": True if toggle is None else toggle,
                    "effort": effort,
                }
        elif isinstance(payload.get("thinking"), (str, bool)):
            payload["thinking"] = {"enabled": payload["thinking"]}
        return payload


class DeepSeekModelPricing(BaseModel):
    input_cost_per_million_tokens: float = Field(ge=0.0)
    output_cost_per_million_tokens: float = Field(ge=0.0)
    cached_input_cost_per_million_tokens: float | None = Field(default=None, ge=0.0)


DEFAULT_DEEPSEEK_MODEL_PRICING: dict[str, DeepSeekModelPricing] = {
    "deepseek-v4-flash": DeepSeekModelPricing(
        input_cost_per_million_tokens=0.14,
        output_cost_per_million_tokens=0.28,
        cached_input_cost_per_million_tokens=0.0028,
    ),
    "deepseek-v4-pro": DeepSeekModelPricing(
        input_cost_per_million_tokens=0.435,
        output_cost_per_million_tokens=0.87,
        cached_input_cost_per_million_tokens=0.003625,
    ),
    "deepseek-chat": DeepSeekModelPricing(
        input_cost_per_million_tokens=0.14,
        output_cost_per_million_tokens=0.28,
        cached_input_cost_per_million_tokens=0.0028,
    ),
    "deepseek-reasoner": DeepSeekModelPricing(
        input_cost_per_million_tokens=0.14,
        output_cost_per_million_tokens=0.28,
        cached_input_cost_per_million_tokens=0.0028,
    ),
}


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    deepseek_api_key: str = Field(min_length=1)
    deepseek_base_url: str = Field(default="https://api.deepseek.com")
    chat_model: str = Field(default="deepseek-v4-flash")
    chat_profile: DeepSeekRequestProfile = Field(default_factory=DeepSeekRequestProfile)
    generation_chat_model: str | None = None
    generation_chat_profile: DeepSeekRequestProfile = Field(default_factory=DeepSeekRequestProfile)
    reflection_chat_model: str | None = None
    reflection_chat_profile: DeepSeekRequestProfile = Field(default_factory=DeepSeekRequestProfile)
    market_volume_estimation_chat_model: str | None = None
    market_volume_estimation_chat_profile: DeepSeekRequestProfile = Field(default_factory=DeepSeekRequestProfile)
    planning_chat_model: str | None = None
    planning_chat_profile: DeepSeekRequestProfile = Field(default_factory=DeepSeekRequestProfile)
    ranking_chat_model: str | None = None
    ranking_chat_profile: DeepSeekRequestProfile = Field(default_factory=DeepSeekRequestProfile)
    evolution_chat_model: str | None = None
    evolution_chat_profile: DeepSeekRequestProfile = Field(default_factory=DeepSeekRequestProfile)
    proximity_chat_model: str | None = None
    proximity_chat_profile: DeepSeekRequestProfile = Field(default_factory=DeepSeekRequestProfile)
    meta_review_chat_model: str | None = None
    meta_review_chat_profile: DeepSeekRequestProfile = Field(default_factory=DeepSeekRequestProfile)
    exa_api_key: str = Field(min_length=1)
    hf_token: str | None = None
    data_dir: Path = Field(default=Path("data"))
    lancedb_path: Path | None = Field(default=None)
    private_graph_path: Path | None = None
    session_decryption_key: bytes | None = None
    embedding_model: str = Field(default="BAAI/bge-base-en-v1.5")
    request_timeout_seconds: int = Field(default=20, ge=5, le=600)
    user_agent: str = Field(default="bmscientist/0.9.4")
    min_relevance_score: float = Field(default=0.6, ge=0.0, le=1.0)
    min_page_characters: int = Field(default=600, ge=100)
    min_snippet_characters: int = Field(default=120, ge=20)
    exa_search_content_text_chars: int = Field(default=8000, ge=500, le=50000)
    exa_contents_initial_text_chars: int = Field(default=12000, ge=500, le=100000)
    exa_contents_deep_text_chars: int = Field(default=50000, ge=1000, le=200000)
    exa_highlights_max_chars: int = Field(default=2000, ge=100, le=10000)
    exa_enable_search_contents: bool = True
    exa_enable_contents_followup: bool = True
    exa_enable_direct_fetch_fallback: bool = True
    exa_default_search_type: str = Field(default="auto")
    exa_reflection_search_type: str = Field(default="fast")
    exa_default_max_age_hours: int = Field(default=168, ge=1, le=24 * 90)
    exa_news_max_age_hours: int = Field(default=24, ge=1, le=24 * 30)
    exa_deep_fetch_min_score: float = Field(default=0.78, ge=0.0, le=1.0)
    exa_deep_fetch_max_per_query: int = Field(default=2, ge=0, le=25)
    exa_deep_fetch_max_per_run: int = Field(default=10, ge=0, le=100)
    exa_search_category: str | None = None
    exa_news_domains: list[str] = Field(default_factory=list)
    deepseek_model_pricing: dict[str, DeepSeekModelPricing] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _derive_lancedb_path(self) -> "AppConfig":
        """Default *lancedb_path* to ``data_dir / 'lancedb'`` when not set."""
        if self.lancedb_path is None:
            self.lancedb_path = self.data_dir / "lancedb"
        return self

    @field_validator("session_decryption_key", mode="before")
    @classmethod
    def _normalize_session_decryption_key(cls, value: Any) -> bytes | None:
        return _coerce_optional_secret_bytes(value)

    @classmethod
    def from_env(cls, env_file: str | Path | None = None) -> "AppConfig":
        load_dotenv(env_file)

        lancedb_env = os.getenv("LANCEDB_PATH")

        values: dict = {
            "deepseek_api_key": os.getenv("DEEPSEEK_API_KEY", ""),
            "deepseek_base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            "chat_model": os.getenv("CHAT_MODEL", "deepseek-v4-flash"),
            "chat_profile": os.getenv("CHAT_PROFILE") or {},
            "generation_chat_model": os.getenv("GENERATION_CHAT_MODEL") or None,
            "generation_chat_profile": os.getenv("GENERATION_CHAT_PROFILE") or {},
            "reflection_chat_model": os.getenv("REFLECTION_CHAT_MODEL") or None,
            "reflection_chat_profile": os.getenv("REFLECTION_CHAT_PROFILE") or {},
            "market_volume_estimation_chat_model": os.getenv("MARKET_VOLUME_ESTIMATION_CHAT_MODEL") or None,
            "market_volume_estimation_chat_profile": os.getenv("MARKET_VOLUME_ESTIMATION_CHAT_PROFILE") or {},
            "planning_chat_model": os.getenv("PLANNING_CHAT_MODEL") or None,
            "planning_chat_profile": os.getenv("PLANNING_CHAT_PROFILE") or {},
            "ranking_chat_model": os.getenv("RANKING_CHAT_MODEL") or None,
            "ranking_chat_profile": os.getenv("RANKING_CHAT_PROFILE") or {},
            "evolution_chat_model": os.getenv("EVOLUTION_CHAT_MODEL") or None,
            "evolution_chat_profile": os.getenv("EVOLUTION_CHAT_PROFILE") or {},
            "proximity_chat_model": os.getenv("PROXIMITY_CHAT_MODEL") or None,
            "proximity_chat_profile": os.getenv("PROXIMITY_CHAT_PROFILE") or {},
            "meta_review_chat_model": os.getenv("META_REVIEW_CHAT_MODEL") or None,
            "meta_review_chat_profile": os.getenv("META_REVIEW_CHAT_PROFILE") or {},
            "exa_api_key": os.getenv("EXA_API_KEY", ""),
            "hf_token": os.getenv("HF_TOKEN") or None,
            "data_dir": Path(os.getenv("BMSCIENTIST_DATA_DIR") or os.getenv("DATA_DIR", "data")),
            "embedding_model": os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5"),
            "request_timeout_seconds": int(os.getenv("REQUEST_TIMEOUT_SECONDS", "60")),
            "exa_search_content_text_chars": int(os.getenv("EXA_SEARCH_CONTENT_TEXT_CHARS", "8000")),
            "exa_contents_initial_text_chars": int(os.getenv("EXA_CONTENTS_INITIAL_TEXT_CHARS", "12000")),
            "exa_contents_deep_text_chars": int(os.getenv("EXA_CONTENTS_DEEP_TEXT_CHARS", "50000")),
            "exa_highlights_max_chars": int(os.getenv("EXA_HIGHLIGHTS_MAX_CHARS", "2000")),
            "exa_enable_search_contents": os.getenv("EXA_ENABLE_SEARCH_CONTENTS", "true"),
            "exa_enable_contents_followup": os.getenv("EXA_ENABLE_CONTENTS_FOLLOWUP", "true"),
            "exa_enable_direct_fetch_fallback": os.getenv("EXA_ENABLE_DIRECT_FETCH_FALLBACK", "true"),
            "exa_default_search_type": os.getenv("EXA_DEFAULT_SEARCH_TYPE", "auto"),
            "exa_reflection_search_type": os.getenv("EXA_REFLECTION_SEARCH_TYPE", "fast"),
            "exa_default_max_age_hours": int(os.getenv("EXA_DEFAULT_MAX_AGE_HOURS", "168")),
            "exa_news_max_age_hours": int(os.getenv("EXA_NEWS_MAX_AGE_HOURS", "24")),
            "exa_deep_fetch_min_score": float(os.getenv("EXA_DEEP_FETCH_MIN_SCORE", "0.78")),
            "exa_deep_fetch_max_per_query": int(os.getenv("EXA_DEEP_FETCH_MAX_PER_QUERY", "2")),
            "exa_deep_fetch_max_per_run": int(os.getenv("EXA_DEEP_FETCH_MAX_PER_RUN", "10")),
            "exa_search_category": os.getenv("EXA_SEARCH_CATEGORY") or None,
            "exa_news_domains": [
                item.strip().lower()
                for item in os.getenv("EXA_NEWS_DOMAINS", "").split(",")
                if item.strip()
            ],
        }
        deepseek_model_pricing_env = os.getenv("DEEPSEEK_MODEL_PRICING")
        if deepseek_model_pricing_env:
            values["deepseek_model_pricing"] = json.loads(deepseek_model_pricing_env)

        if lancedb_env:
            values["lancedb_path"] = Path(lancedb_env)
        private_graph_path_env = os.getenv("PRIVATE_GRAPH_PATH")
        if private_graph_path_env:
            values["private_graph_path"] = Path(private_graph_path_env)
        session_decryption_key_env = os.getenv("SESSION_DECRYPTION_KEY")
        if session_decryption_key_env:
            values["session_decryption_key"] = session_decryption_key_env

        from pydantic import ValidationError
        try:
            return cls.model_validate(values)
        except ValidationError as e:
            invalid_fields = []
            for err in e.errors():
                loc = err.get("loc", ())
                if loc:
                    invalid_fields.append(str(loc[0]))
            if invalid_fields:
                raise RuntimeError(
                    f"Invalid or missing configuration keys: {', '.join(invalid_fields)}. "
                    "Please check that your backend/.env file contains all required variables with valid values."
                )
            raise

    @model_validator(mode="after")
    def _merge_chat_profiles(self) -> "AppConfig":
        self.chat_profile = self._resolved_profile(self.chat_profile, self.chat_model)
        self.generation_chat_profile = self._resolved_profile(self.generation_chat_profile, self.generation_chat_model)
        self.reflection_chat_profile = self._resolved_profile(self.reflection_chat_profile, self.reflection_chat_model)
        self.market_volume_estimation_chat_profile = self._resolved_profile(
            self.market_volume_estimation_chat_profile,
            self.market_volume_estimation_chat_model,
        )
        self.planning_chat_profile = self._resolved_profile(self.planning_chat_profile, self.planning_chat_model)
        self.ranking_chat_profile = self._resolved_profile(self.ranking_chat_profile, self.ranking_chat_model)
        self.evolution_chat_profile = self._resolved_profile(self.evolution_chat_profile, self.evolution_chat_model)
        self.proximity_chat_profile = self._resolved_profile(self.proximity_chat_profile, self.proximity_chat_model)
        self.meta_review_chat_profile = self._resolved_profile(self.meta_review_chat_profile, self.meta_review_chat_model)
        return self

    @staticmethod
    def _resolved_profile(profile: DeepSeekRequestProfile | None, model: str | None) -> DeepSeekRequestProfile:
        if profile is None:
            profile = DeepSeekRequestProfile()
        if model:
            return profile.model_copy(update={"model": model})
        if profile.model:
            return profile
        return profile

    def resolved_lancedb_path(self) -> Path:
        return self.lancedb_path.expanduser().resolve()

    def ensure_directories(self) -> None:
        self.resolved_lancedb_path().mkdir(parents=True, exist_ok=True)
        (self.data_dir / "raw").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "outputs").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "manually-obtained").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "manually-obtained" / "processed").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "coscientist").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "graph").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "pricing").mkdir(parents=True, exist_ok=True)

    def resolve_deepseek_model_pricing(self, model: str | None) -> DeepSeekModelPricing | None:
        if not model:
            return None
        normalized = model.strip()
        if not normalized:
            return None
        if normalized in self.deepseek_model_pricing:
            return self.deepseek_model_pricing[normalized]
        return DEFAULT_DEEPSEEK_MODEL_PRICING.get(normalized)
