import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List

from config import LLM_CONFIG


@dataclass
class LLMConfig:
    provider: str = "openai_compatible"
    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    timeout: float = 30.0
    temperature: float = 0.0
    max_tokens: int = 256

    @classmethod
    def from_settings(cls, settings: Dict[str, Any]) -> "LLMConfig":
        provider = str(settings.get("provider", cls.provider)).strip().lower()
        base_url = str(settings.get("base_url", cls.base_url)).strip()
        api_key = str(settings.get("api_key", cls.api_key))
        model = str(settings.get("model", cls.model)).strip()
        timeout = float(settings.get("timeout", cls.timeout))
        temperature = float(settings.get("temperature", cls.temperature))
        max_tokens = int(settings.get("max_tokens", cls.max_tokens))

        if not provider:
            provider = "openai_compatible"
            if "ollama" in base_url:
                provider = "ollama"

        if provider == "lmstudio":
            provider = "openai_compatible"

        return cls(
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    @classmethod
    def from_env(cls) -> "LLMConfig":
        settings = dict(LLM_CONFIG)
        settings["provider"] = os.getenv("LLM_PROVIDER", os.getenv("LLM_BACKEND", settings["provider"]))
        settings["base_url"] = os.getenv("LLM_BASE_URL", settings["base_url"])
        settings["api_key"] = os.getenv("LLM_API_KEY", settings["api_key"])
        settings["model"] = os.getenv("LLM_MODEL", settings["model"])
        settings["timeout"] = os.getenv("LLM_TIMEOUT", settings["timeout"])
        settings["temperature"] = os.getenv("LLM_TEMPERATURE", settings["temperature"])
        settings["max_tokens"] = os.getenv("LLM_MAX_TOKENS", settings["max_tokens"])
        return cls.from_settings(settings)


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig.from_env()

    def chat(self, messages: List[Dict[str, str]]) -> str:
        provider = self.config.provider
        if provider in {"openai", "openai_compatible", "lmstudio"}:
            return self._chat_openai_compatible(messages)
        if provider == "ollama":
            return self._chat_ollama(messages)
        raise LLMError(f"Unsupported LLM provider: {provider}")

    def _chat_openai_compatible(self, messages: List[Dict[str, str]]) -> str:
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        response = self._post_json(url, payload, headers)
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected response shape from provider: {response}") from exc

    def _chat_ollama(self, messages: List[Dict[str, str]]) -> str:
        url = self.config.base_url.rstrip("/") + "/api/chat"
        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        }
        response = self._post_json(url, payload, {"Content-Type": "application/json"})
        try:
            return response["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise LLMError(f"Unexpected response shape from Ollama: {response}") from exc

    def _post_json(self, url: str, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LLMError(f"LLM HTTP error {exc.code} from {url}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise LLMError(f"Could not connect to LLM provider at {url}: {exc.reason}") from exc

        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMError(f"LLM provider returned invalid JSON: {raw}") from exc
