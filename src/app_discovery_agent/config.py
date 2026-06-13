from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    deepseek_api_key: str = Field(min_length=1)
    deepseek_base_url: str = Field(default="https://api.deepseek.com")
    chat_model: str = Field(default="deepseek-v4-flash")
    generation_chat_model: str | None = None
    reflection_chat_model: str | None = None
    planning_chat_model: str | None = None
    ranking_chat_model: str | None = None
    evolution_chat_model: str | None = None
    proximity_chat_model: str | None = None
    meta_review_chat_model: str | None = None
    exa_api_key: str = Field(min_length=1)
    lancedb_path: Path = Field(default=Path("./data/lancedb"))
    embedding_model: str = Field(default="BAAI/bge-base-en-v1.5")
    request_timeout_seconds: int = Field(default=20, ge=5, le=600)
    user_agent: str = Field(default="app-discovery-agent/0.1.0")
    min_relevance_score: float = Field(default=0.6, ge=0.0, le=1.0)
    min_page_characters: int = Field(default=600, ge=100)
    min_snippet_characters: int = Field(default=120, ge=20)
    skip_fetch_domains: list[str] = Field(default_factory=lambda: ["sciencedirect.com"])

    @classmethod
    def from_env(cls, env_file: str | Path | None = None) -> "AppConfig":
        load_dotenv(env_file)
        return cls.model_validate(
            {
                "deepseek_api_key": os.getenv("DEEPSEEK_API_KEY", ""),
                "deepseek_base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                "chat_model": os.getenv("CHAT_MODEL", "deepseek-v4-flash"),
                "generation_chat_model": os.getenv("GENERATION_CHAT_MODEL") or None,
                "reflection_chat_model": os.getenv("REFLECTION_CHAT_MODEL") or None,
                "planning_chat_model": os.getenv("PLANNING_CHAT_MODEL") or None,
                "ranking_chat_model": os.getenv("RANKING_CHAT_MODEL") or None,
                "evolution_chat_model": os.getenv("EVOLUTION_CHAT_MODEL") or None,
                "proximity_chat_model": os.getenv("PROXIMITY_CHAT_MODEL") or None,
                "meta_review_chat_model": os.getenv("META_REVIEW_CHAT_MODEL") or None,
                "exa_api_key": os.getenv("EXA_API_KEY", ""),
                "lancedb_path": Path(os.getenv("LANCEDB_PATH", "./data/lancedb")),
                "embedding_model": os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5"),
                "request_timeout_seconds": int(os.getenv("REQUEST_TIMEOUT_SECONDS", "60")),
                "skip_fetch_domains": [
                    item.strip()
                    for item in os.getenv("SKIP_FETCH_DOMAINS", "sciencedirect.com").split(",")
                    if item.strip()
                ],
            }
        )

    def resolved_lancedb_path(self) -> Path:
        return self.lancedb_path.expanduser().resolve()

    def ensure_directories(self) -> None:
        self.resolved_lancedb_path().mkdir(parents=True, exist_ok=True)
        Path("data/raw").mkdir(parents=True, exist_ok=True)
        Path("data/outputs").mkdir(parents=True, exist_ok=True)
        Path("data/manually-obtained").mkdir(parents=True, exist_ok=True)
        Path("data/manually-obtained/processed").mkdir(parents=True, exist_ok=True)
        Path("data/coscientist").mkdir(parents=True, exist_ok=True)
