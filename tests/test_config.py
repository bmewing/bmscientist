from pathlib import Path

from app_discovery_agent.config import AppConfig


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
