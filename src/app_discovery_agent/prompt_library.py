from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from string import Template
from threading import Lock


@dataclass
class _PromptCacheEntry:
    modified_time_ns: int
    sections: dict[str, str]


class PromptLibrary:
    def __init__(self, base_dir: Path | None = None):
        self._base_dir = base_dir or Path(__file__).resolve().parents[2] / "prompts" / "agents"
        self._cache: dict[Path, _PromptCacheEntry] = {}
        self._lock = Lock()

    def render(self, agent_name: str, section_name: str, **context: object) -> str:
        sections = self._load_sections(agent_name)
        if section_name not in sections:
            raise KeyError(f"Prompt section '{section_name}' not found in '{agent_name}.md'.")
        return Template(sections[section_name]).substitute(context)

    def _load_sections(self, agent_name: str) -> dict[str, str]:
        path = self._base_dir / f"{agent_name}.md"
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")

        modified_time_ns = path.stat().st_mtime_ns
        with self._lock:
            cached = self._cache.get(path)
            if cached is not None and cached.modified_time_ns == modified_time_ns:
                return cached.sections

            sections = self._parse_sections(path.read_text(encoding="utf-8"))
            self._cache[path] = _PromptCacheEntry(modified_time_ns=modified_time_ns, sections=sections)
            return sections

    @staticmethod
    def _parse_sections(content: str) -> dict[str, str]:
        sections: dict[str, str] = {}
        current_section: str | None = None
        buffer: list[str] = []

        def flush() -> None:
            nonlocal buffer
            if current_section is None:
                buffer = []
                return
            sections[current_section] = "\n".join(buffer).strip()
            buffer = []

        for line in content.splitlines():
            if line.startswith("## "):
                flush()
                current_section = line[3:].strip()
                continue
            if current_section is not None:
                buffer.append(line)

        flush()
        return sections


PROMPTS = PromptLibrary()
