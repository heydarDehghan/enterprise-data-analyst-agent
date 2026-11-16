"""Validated LLM settings from the process environment and project .env."""

import os
from collections import ChainMap
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from dotenv.parser import parse_stream


_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
_DEFAULT_URL = httpx.URL("https://openrouter.ai/api/v1/chat/completions")
_DEFAULT_MODEL = "qwen/qwen3.8-27b"


class LLMConfigurationError(RuntimeError):
    """A configuration error whose message contains no setting values."""


@dataclass(frozen=True)
class LLMConfig:
    url: str
    model: str
    api_key: str = field(repr=False)
    max_completion_tokens: int | None


def _environment() -> ChainMap[str, str]:
    file_values: dict[str, str] = {}
    try:
        with _ENV_FILE.open(encoding="utf-8-sig") as stream:
            for binding in parse_stream(stream):
                if binding.error:
                    raise LLMConfigurationError(f"Invalid .env syntax at line {binding.original.line}")
                if binding.key is not None:
                    file_values[binding.key] = binding.value or ""
    except FileNotFoundError:
        pass
    except (OSError, UnicodeError):
        raise LLMConfigurationError("Unable to read the project .env as UTF-8") from None
    return ChainMap(os.environ, file_values)


def _positive_limit(
    name: str, value: str | None, *, default: int | None, ceiling: int,
) -> int | None:
    if value is None:
        return default
    try:
        limit = int(value)
    except ValueError:
        limit = None
    if limit is None or not 1 <= limit <= ceiling:
        raise LLMConfigurationError(f"{name} must be an integer from 1 to {ceiling}")
    return limit


def load_llm_config() -> LLMConfig:
    values = _environment()
    url = values.get("LLM_CHAT_COMPLETIONS_URL", str(_DEFAULT_URL))
    if not url or not url.isprintable() or " " in url or "\\" in url:
        raise LLMConfigurationError("LLM_CHAT_COMPLETIONS_URL must be a valid HTTPS URL")
    try:
        endpoint = httpx.URL(url)
    except (httpx.InvalidURL, ValueError):
        raise LLMConfigurationError("LLM_CHAT_COMPLETIONS_URL must be a valid HTTPS URL") from None
    port = endpoint.port
    if (endpoint.scheme != "https" or not endpoint.host or endpoint.userinfo
            or "#" in url or (port is not None and not 1 <= port <= 65535)):
        raise LLMConfigurationError(
            "LLM_CHAT_COMPLETIONS_URL must be a valid HTTPS URL without credentials or a fragment"
        )

    model = values.get("LLM_MODEL", _DEFAULT_MODEL)
    if not model or not model.strip() or not model.isprintable():
        raise LLMConfigurationError("LLM_MODEL must contain a valid model name")

    api_key = values.get("LLM_API_KEY")
    if not api_key or not api_key.strip():
        raise LLMConfigurationError("Set LLM_API_KEY in the environment or project .env")
    if not api_key.isascii() or not api_key.isprintable() or " " in api_key:
        raise LLMConfigurationError("LLM_API_KEY must contain ASCII characters without whitespace")

    # Keep the token field optional on custom endpoints for legacy providers.
    completion_limit = _positive_limit(
        "LLM_MAX_COMPLETION_TOKENS",
        values.get("LLM_MAX_COMPLETION_TOKENS"),
        default=2048 if endpoint == _DEFAULT_URL else None,
        ceiling=8192,
    )
    return LLMConfig(url, model, api_key, completion_limit)


def load_llm_budget() -> tuple[int | None, int | None]:
    values = _environment()
    calls = _positive_limit(
        "LLM_MAX_CALLS_PER_REQUEST",
        values.get("LLM_MAX_CALLS_PER_REQUEST"),
        default=12,
        ceiling=32,
    )
    tokens = _positive_limit(
        "LLM_MAX_TOTAL_TOKENS_PER_REQUEST",
        values.get("LLM_MAX_TOTAL_TOKENS_PER_REQUEST"),
        default=None,
        ceiling=10_000_000,
    )
    return calls, tokens
