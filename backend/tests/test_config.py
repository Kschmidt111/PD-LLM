from pd_assistant.core.config import Settings, get_settings

_ENV_KEYS = (
    "APP_ENV",
    "LOG_LEVEL",
    "TOP_K",
    "CHUNK_SIZE",
    "OLLAMA_BASE_URL",
    "OLLAMA_MODEL",
)


def test_settings_defaults_and_env_override(monkeypatch) -> None:
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    get_settings.cache_clear()

    settings = Settings()
    assert settings.app_env == "dev"
    assert settings.log_level == "INFO"
    assert settings.top_k == 5
    assert settings.chunk_size == 800
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.ollama_model == "llama3"

    monkeypatch.setenv("TOP_K", "8")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.1")
    get_settings.cache_clear()

    overridden = Settings()
    assert overridden.top_k == 8
    assert overridden.ollama_model == "llama3.1"
