"""
Global project settings.

Edit the LLM_CONFIG block below when you want to switch providers or endpoints.
Environment variables still work as overrides, but this file is the single
source of truth for local development and shared runs.

Examples:

- LM Studio / any OpenAI-compatible local server:
  provider = "openai_compatible"
  base_url = "http://localhost:1234/v1"
  api_key = "lm-studio"
  model = "deepseek-coder-33b-instruct"

- Ollama:
  provider = "ollama"
  base_url = "http://localhost:11434"
  api_key = ""
  model = "llama3.1"

- Direct OpenAI API:
  provider = "openai"
  base_url = "https://api.openai.com/v1"
  api_key = "your-openai-key"
  model = "gpt-4o-mini"
"""

LLM_CONFIG = {
    "provider": "openai_compatible",
    "base_url": "http://localhost:1234/v1",
    "api_key": "lm-studio",
    "model": "deepseek-coder-33b-instruct",
    "timeout": 30.0,
    "temperature": 0.0,
    "max_tokens": 256,
}

SEARCH_CONFIG = {
    "top_k": 3,
    "context_chars": 800,
}
