from types import SimpleNamespace

import pytest

from bmscientist.config import AppConfig
from bmscientist.cost_tracking import CostTracker


def test_cost_tracker_computes_provider_totals():
    config = AppConfig(
        deepseek_api_key="x",
        exa_api_key="y",
    )
    tracker = CostTracker(config)

    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500,
            prompt_cache_hit_tokens=200,
            prompt_cache_miss_tokens=800,
        )
    )

    tracker.record_deepseek_response(
        response=response,
        model="deepseek-v4-flash",
        client_name="reflection",
    )
    tracker.record_exa_call(operation="search", cost_dollars=0.12)
    tracker.record_exa_call(operation="contents", cost_dollars=0.03)

    report = tracker.build_report("research-1")

    assert report["total_exa_usd"] == pytest.approx(0.15)
    assert report["total_deepseek_usd"] == pytest.approx(0.000253, abs=1e-6)
    assert report["providers"]["exa"]["by_operation"]["search"]["total_usd"] == pytest.approx(0.12)
    assert report["providers"]["deepseek"]["by_model"]["deepseek-v4-flash"]["cached_prompt_tokens"] == 200
    assert report["providers"]["deepseek"]["by_client"]["reflection"]["completion_tokens"] == 500


def test_cost_tracker_can_seed_from_existing_report():
    config = AppConfig(
        deepseek_api_key="x",
        exa_api_key="y",
    )
    tracker = CostTracker(config)
    tracker.seed_from_report(
        {
            "providers": {
                "exa": {
                    "calls": 2,
                    "total_usd": 0.5,
                    "by_operation": {
                        "search": {"calls": 1, "total_usd": 0.3},
                        "contents": {"calls": 1, "total_usd": 0.2},
                    },
                },
                "deepseek": {
                    "calls": 1,
                    "priced_calls": 1,
                    "unpriced_calls": 0,
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                    "cached_prompt_tokens": 10,
                    "total_usd": 0.01,
                    "by_model": {
                        "deepseek-v4-pro": {
                            "calls": 1,
                            "priced_calls": 1,
                            "unpriced_calls": 0,
                            "prompt_tokens": 100,
                            "completion_tokens": 50,
                            "total_tokens": 150,
                            "cached_prompt_tokens": 10,
                            "total_usd": 0.01,
                        }
                    },
                    "by_client": {
                        "planning": {
                            "calls": 1,
                            "priced_calls": 1,
                            "unpriced_calls": 0,
                            "prompt_tokens": 100,
                            "completion_tokens": 50,
                            "total_tokens": 150,
                            "cached_prompt_tokens": 10,
                            "total_usd": 0.01,
                        }
                    },
                },
            }
        }
    )

    report = tracker.build_report("research-1")

    assert report["total_exa_usd"] == pytest.approx(0.5)
    assert report["total_deepseek_usd"] == pytest.approx(0.01)
    assert report["providers"]["deepseek"]["by_model"]["deepseek-v4-pro"]["prompt_tokens"] == 100
