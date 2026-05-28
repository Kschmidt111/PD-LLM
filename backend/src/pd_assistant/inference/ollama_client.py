"""LLM client for Ollama."""

# read settings from get_settings() where I alreasdy have ollama_base_url and ollama_model

from pd_assistant.core.config import get_settings
from pd_assistant.inference.interfaces import ChatMessage, LLMResponse
import httpx

class OllamaClient:
    def __init__(self, base_url: str = None, model: str = None) -> None:
        if base_url is None:
            base_url = get_settings().ollama_base_url
        
        if model is None:
            model = get_settings().ollama_model

        self._base_url = base_url
        self._model = model


    def health_check(self) -> bool:
        try:
            response = httpx.get(f"{self._base_url}/api/tags", timeout=5.0)
            if response.status_code == 200:
                return True
            else:
                return False
        
        except httpx.HTTPError as e:
            return False



    def generate(self, messages: list[ChatMessage]) -> LLMResponse:
        return NotImplementedError("generate is not implemented")

        
