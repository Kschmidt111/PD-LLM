import pytest
from pd_assistant.inference.ollama_client import OllamaClient
from pd_assistant.inference.interfaces import LLMClient, ChatMessage, LLMResponse

@pytest.mark.integration
def test_ollama_client_health_check() -> None:
    client = OllamaClient()
    
    if not client.health_check():
        pytest.skip("Ollama is not running")
    else:
        assert client.health_check() is True