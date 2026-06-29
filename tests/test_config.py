import pytest
from pathlib import Path

from bmscientist.config import AppConfig


def test_config_loads_hf_token_from_env_file(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "DEEPSEEK_API_KEY=x",
                "EXA_API_KEY=y",
                "HF_TOKEN=hf_test_token",
            ]
        ),
        encoding="utf-8",
    )

    config = AppConfig.from_env(env_path)

    assert config.hf_token == "hf_test_token"


def test_config_missing_keys_throws_runtime_error(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("EXA_API_KEY", raising=False)

    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "DEEPSEEK_API_KEY=",
                "EXA_API_KEY=",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError) as exc_info:
        AppConfig.from_env(env_path)

    assert "Invalid or missing configuration keys" in str(exc_info.value)
    assert "deepseek_api_key" in str(exc_info.value)


def test_config_loads_chat_profile_and_normalizes_thinking_effort(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "DEEPSEEK_API_KEY=x",
                "EXA_API_KEY=y",
                'REFLECTION_CHAT_PROFILE={"model":"deepseek-v4-pro","thinking":{"enabled":"yes","effort":"xhigh"}}',
            ]
        ),
        encoding="utf-8",
    )

    config = AppConfig.from_env(env_path)

    assert config.reflection_chat_profile.model == "deepseek-v4-pro"
    assert config.reflection_chat_profile.thinking is not None
    assert config.reflection_chat_profile.thinking.enabled is True
    assert config.reflection_chat_profile.thinking.effort == "max"


def test_config_loads_profile_specific_timeout(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "DEEPSEEK_API_KEY=x",
                "EXA_API_KEY=y",
                'MARKET_VOLUME_ESTIMATION_CHAT_PROFILE={"model":"deepseek-v4-pro","thinking":{"enabled":true,"effort":"max"},"timeout_seconds":180}',
            ]
        ),
        encoding="utf-8",
    )

    config = AppConfig.from_env(env_path)

    assert config.market_volume_estimation_chat_profile.model == "deepseek-v4-pro"
    assert config.market_volume_estimation_chat_profile.timeout_seconds == 180


def test_config_loads_exa_retrieval_settings(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "DEEPSEEK_API_KEY=x",
                "EXA_API_KEY=y",
                "EXA_SEARCH_CONTENT_TEXT_CHARS=9000",
                "EXA_CONTENTS_INITIAL_TEXT_CHARS=15000",
                "EXA_REFLECTION_SEARCH_TYPE=auto",
                "EXA_NEWS_DOMAINS=polymart.info,icis.com",
            ]
        ),
        encoding="utf-8",
    )

    config = AppConfig.from_env(env_path)

    assert config.exa_search_content_text_chars == 9000
    assert config.exa_contents_initial_text_chars == 15000
    assert config.exa_reflection_search_type == "auto"
    assert config.exa_news_domains == ["polymart.info", "icis.com"]


def test_config_loads_deepseek_model_pricing_override(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "DEEPSEEK_API_KEY=x",
                "EXA_API_KEY=y",
                'DEEPSEEK_MODEL_PRICING={"deepseek-v4-pro":{"input_cost_per_million_tokens":1.0,"output_cost_per_million_tokens":2.0,"cached_input_cost_per_million_tokens":0.1}}',
            ]
        ),
        encoding="utf-8",
    )

    config = AppConfig.from_env(env_path)

    pricing = config.resolve_deepseek_model_pricing("deepseek-v4-pro")
    assert pricing is not None
    assert pricing.input_cost_per_million_tokens == 1.0
    assert pricing.output_cost_per_million_tokens == 2.0
    assert pricing.cached_input_cost_per_million_tokens == 0.1


def test_config_loads_private_graph_settings_and_session_key(tmp_path):
    session_key_hex = "11" * 32
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "DEEPSEEK_API_KEY=x",
                "EXA_API_KEY=y",
                "PRIVATE_GRAPH_PATH=data/private-graph",
                f"SESSION_DECRYPTION_KEY={session_key_hex}",
            ]
        ),
        encoding="utf-8",
    )

    config = AppConfig.from_env(env_path)

    assert config.private_graph_path == Path("data/private-graph")
    assert config.session_decryption_key == bytes.fromhex(session_key_hex)
