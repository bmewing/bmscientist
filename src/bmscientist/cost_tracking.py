from __future__ import annotations

from datetime import datetime, timezone
from threading import Lock
from typing import Any

from bmscientist.config import AppConfig


class CostTracker:
    def __init__(self, config: AppConfig):
        self._config = config
        self._lock = Lock()
        self._exa_total_usd = 0.0
        self._exa_calls = 0
        self._exa_by_operation: dict[str, dict[str, float | int]] = {}
        self._deepseek_total_usd = 0.0
        self._deepseek_calls = 0
        self._deepseek_priced_calls = 0
        self._deepseek_unpriced_calls = 0
        self._deepseek_prompt_tokens = 0
        self._deepseek_completion_tokens = 0
        self._deepseek_total_tokens = 0
        self._deepseek_cached_prompt_tokens = 0
        self._deepseek_by_model: dict[str, dict[str, float | int]] = {}
        self._deepseek_by_client: dict[str, dict[str, float | int]] = {}

    def record_exa_call(
        self,
        *,
        operation: str,
        cost_dollars: float | None,
    ) -> None:
        with self._lock:
            self._exa_calls += 1
            amount = float(cost_dollars or 0.0)
            self._exa_total_usd += amount
            bucket = self._exa_by_operation.setdefault(operation, {"calls": 0, "total_usd": 0.0})
            bucket["calls"] = int(bucket["calls"]) + 1
            bucket["total_usd"] = float(bucket["total_usd"]) + amount

    def record_deepseek_response(
        self,
        *,
        response: Any,
        model: str,
        client_name: str | None = None,
    ) -> None:
        usage = getattr(response, "usage", None)
        prompt_tokens = _usage_value(usage, "prompt_tokens")
        completion_tokens = _usage_value(usage, "completion_tokens")
        total_tokens = _usage_value(usage, "total_tokens")
        cached_prompt_tokens = _usage_nested_value(usage, "prompt_tokens_details", "cached_tokens")
        if cached_prompt_tokens is None:
            cached_prompt_tokens = _usage_value(usage, "prompt_cache_hit_tokens")
        prompt_miss_tokens = _usage_value(usage, "prompt_cache_miss_tokens")
        if prompt_miss_tokens is None and prompt_tokens is not None and cached_prompt_tokens is not None:
            prompt_miss_tokens = max(prompt_tokens - cached_prompt_tokens, 0)
        if prompt_tokens is None and prompt_miss_tokens is not None and cached_prompt_tokens is not None:
            prompt_tokens = prompt_miss_tokens + cached_prompt_tokens
        if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens

        pricing = self._config.resolve_deepseek_model_pricing(model)
        total_cost = None
        if pricing is not None:
            prompt_cost = 0.0
            if prompt_miss_tokens is not None:
                prompt_cost += (prompt_miss_tokens / 1_000_000) * pricing.input_cost_per_million_tokens
            elif prompt_tokens is not None:
                prompt_cost += (prompt_tokens / 1_000_000) * pricing.input_cost_per_million_tokens
            if cached_prompt_tokens is not None:
                cached_rate = (
                    pricing.cached_input_cost_per_million_tokens
                    if pricing.cached_input_cost_per_million_tokens is not None
                    else pricing.input_cost_per_million_tokens
                )
                prompt_cost += (cached_prompt_tokens / 1_000_000) * cached_rate
            output_cost = (
                (completion_tokens / 1_000_000) * pricing.output_cost_per_million_tokens
                if completion_tokens is not None
                else 0.0
            )
            total_cost = prompt_cost + output_cost

        with self._lock:
            self._deepseek_calls += 1
            self._deepseek_prompt_tokens += int(prompt_tokens or 0)
            self._deepseek_completion_tokens += int(completion_tokens or 0)
            self._deepseek_total_tokens += int(total_tokens or 0)
            self._deepseek_cached_prompt_tokens += int(cached_prompt_tokens or 0)
            if total_cost is None:
                self._deepseek_unpriced_calls += 1
            else:
                self._deepseek_priced_calls += 1
                self._deepseek_total_usd += total_cost

            model_bucket = self._deepseek_by_model.setdefault(model, _deepseek_bucket())
            _update_deepseek_bucket(
                model_bucket,
                total_cost=total_cost,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cached_prompt_tokens=cached_prompt_tokens,
            )

            bucket_name = client_name or model
            client_bucket = self._deepseek_by_client.setdefault(bucket_name, _deepseek_bucket())
            _update_deepseek_bucket(
                client_bucket,
                total_cost=total_cost,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cached_prompt_tokens=cached_prompt_tokens,
            )

    def seed_from_report(self, report: dict[str, Any]) -> None:
        providers = report.get("providers", {}) if isinstance(report, dict) else {}
        exa = providers.get("exa", {})
        deepseek = providers.get("deepseek", {})
        with self._lock:
            self._exa_total_usd += float(exa.get("total_usd", 0.0) or 0.0)
            self._exa_calls += int(exa.get("calls", 0) or 0)
            for operation, stats in (exa.get("by_operation", {}) or {}).items():
                bucket = self._exa_by_operation.setdefault(operation, {"calls": 0, "total_usd": 0.0})
                bucket["calls"] = int(bucket["calls"]) + int((stats or {}).get("calls", 0) or 0)
                bucket["total_usd"] = float(bucket["total_usd"]) + float((stats or {}).get("total_usd", 0.0) or 0.0)

            self._deepseek_total_usd += float(deepseek.get("total_usd", 0.0) or 0.0)
            self._deepseek_calls += int(deepseek.get("calls", 0) or 0)
            self._deepseek_priced_calls += int(deepseek.get("priced_calls", 0) or 0)
            self._deepseek_unpriced_calls += int(deepseek.get("unpriced_calls", 0) or 0)
            self._deepseek_prompt_tokens += int(deepseek.get("prompt_tokens", 0) or 0)
            self._deepseek_completion_tokens += int(deepseek.get("completion_tokens", 0) or 0)
            self._deepseek_total_tokens += int(deepseek.get("total_tokens", 0) or 0)
            self._deepseek_cached_prompt_tokens += int(deepseek.get("cached_prompt_tokens", 0) or 0)

            for model, stats in (deepseek.get("by_model", {}) or {}).items():
                bucket = self._deepseek_by_model.setdefault(model, _deepseek_bucket())
                _merge_deepseek_bucket(bucket, stats or {})
            for client_name, stats in (deepseek.get("by_client", {}) or {}).items():
                bucket = self._deepseek_by_client.setdefault(client_name, _deepseek_bucket())
                _merge_deepseek_bucket(bucket, stats or {})

    def build_report(self, research_id: str) -> dict[str, Any]:
        with self._lock:
            total_exa_usd = round(self._exa_total_usd, 6)
            total_deepseek_usd = round(self._deepseek_total_usd, 6)
            return {
                "research_id": research_id,
                "currency": "USD",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "total_exa_usd": total_exa_usd,
                "total_deepseek_usd": total_deepseek_usd,
                "total_usd": round(total_exa_usd + total_deepseek_usd, 6),
                "providers": {
                    "exa": {
                        "cost_source": "exa costDollars.total",
                        "calls": self._exa_calls,
                        "total_usd": total_exa_usd,
                        "by_operation": {
                            name: {
                                "calls": int(stats["calls"]),
                                "total_usd": round(float(stats["total_usd"]), 6),
                            }
                            for name, stats in sorted(self._exa_by_operation.items())
                        },
                    },
                    "deepseek": {
                        "cost_source": "token usage x configured model pricing",
                        "calls": self._deepseek_calls,
                        "priced_calls": self._deepseek_priced_calls,
                        "unpriced_calls": self._deepseek_unpriced_calls,
                        "prompt_tokens": self._deepseek_prompt_tokens,
                        "completion_tokens": self._deepseek_completion_tokens,
                        "total_tokens": self._deepseek_total_tokens,
                        "cached_prompt_tokens": self._deepseek_cached_prompt_tokens,
                        "total_usd": total_deepseek_usd,
                        "by_model": {
                            name: _rounded_deepseek_bucket(stats)
                            for name, stats in sorted(self._deepseek_by_model.items())
                        },
                        "by_client": {
                            name: _rounded_deepseek_bucket(stats)
                            for name, stats in sorted(self._deepseek_by_client.items())
                        },
                    },
                },
            }


def _usage_value(usage: Any, field: str) -> int | None:
    if usage is None:
        return None
    value = getattr(usage, field, None)
    if value is None and isinstance(usage, dict):
        value = usage.get(field)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _usage_nested_value(usage: Any, field: str, nested_field: str) -> int | None:
    if usage is None:
        return None
    nested = getattr(usage, field, None)
    if nested is None and isinstance(usage, dict):
        nested = usage.get(field)
    if nested is None:
        return None
    value = getattr(nested, nested_field, None)
    if value is None and isinstance(nested, dict):
        value = nested.get(nested_field)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _deepseek_bucket() -> dict[str, float | int]:
    return {
        "calls": 0,
        "priced_calls": 0,
        "unpriced_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cached_prompt_tokens": 0,
        "total_usd": 0.0,
    }


def _update_deepseek_bucket(
    bucket: dict[str, float | int],
    *,
    total_cost: float | None,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    total_tokens: int | None,
    cached_prompt_tokens: int | None,
) -> None:
    bucket["calls"] = int(bucket["calls"]) + 1
    bucket["prompt_tokens"] = int(bucket["prompt_tokens"]) + int(prompt_tokens or 0)
    bucket["completion_tokens"] = int(bucket["completion_tokens"]) + int(completion_tokens or 0)
    bucket["total_tokens"] = int(bucket["total_tokens"]) + int(total_tokens or 0)
    bucket["cached_prompt_tokens"] = int(bucket["cached_prompt_tokens"]) + int(cached_prompt_tokens or 0)
    if total_cost is None:
        bucket["unpriced_calls"] = int(bucket["unpriced_calls"]) + 1
        return
    bucket["priced_calls"] = int(bucket["priced_calls"]) + 1
    bucket["total_usd"] = float(bucket["total_usd"]) + total_cost


def _merge_deepseek_bucket(bucket: dict[str, float | int], incoming: dict[str, Any]) -> None:
    for field in (
        "calls",
        "priced_calls",
        "unpriced_calls",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cached_prompt_tokens",
    ):
        bucket[field] = int(bucket[field]) + int(incoming.get(field, 0) or 0)
    bucket["total_usd"] = float(bucket["total_usd"]) + float(incoming.get("total_usd", 0.0) or 0.0)


def _rounded_deepseek_bucket(bucket: dict[str, float | int]) -> dict[str, float | int]:
    return {
        "calls": int(bucket["calls"]),
        "priced_calls": int(bucket["priced_calls"]),
        "unpriced_calls": int(bucket["unpriced_calls"]),
        "prompt_tokens": int(bucket["prompt_tokens"]),
        "completion_tokens": int(bucket["completion_tokens"]),
        "total_tokens": int(bucket["total_tokens"]),
        "cached_prompt_tokens": int(bucket["cached_prompt_tokens"]),
        "total_usd": round(float(bucket["total_usd"]), 6),
    }
