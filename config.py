"""Settings, read once from the environment (and .env when running locally)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


@dataclass(frozen=True)
class Settings:
    # The name this agent registers under. A phone number's dispatch rule must
    # name it: a named agent only joins calls dispatched to it.
    agent_name: str = field(default_factory=lambda: _env("AGENT_NAME", "dasco-v3"))

    # Deepgram is the agent's voice.
    deepgram_api_key: str = field(default_factory=lambda: _env("DEEPGRAM_API_KEY"))
    tts_model: str = field(default_factory=lambda: _env("DEEPGRAM_TTS_MODEL", "aura-2-thalia-en"))

    # Speech-to-text with speaker diarization, so only the caller's voice is answered.
    speechmatics_api_key: str = field(default_factory=lambda: _env("SPEECHMATICS_API_KEY"))
    speechmatics_language: str = field(
        default_factory=lambda: _env("SPEECHMATICS_LANGUAGE", "en")
    )

    # The LLM: Claude Haiku 5.5, the fastest and cheapest Claude model.
    anthropic_api_key: str = field(default_factory=lambda: _env("ANTHROPIC_API_KEY"))
    claude_model: str = field(default_factory=lambda: _env("LLM_MODEL", "claude-haiku-5-5"))
    # Measured on this agent's prompt: thinking off at low effort starts replying in about
    # a second; with thinking on it took up to six. A ceiling on reply length, not a target.
    claude_effort: str = field(default_factory=lambda: _env("LLM_EFFORT", "low"))
    claude_thinking: bool = field(
        default_factory=lambda: _env("LLM_THINKING", "off").lower() in ("on", "true", "1")
    )
    claude_max_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_TOKENS", 1024))

    dealership_name: str = field(default_factory=lambda: _env("DEALERSHIP_NAME", "PASCO Motors"))
    showroom_open_hour: int = field(default_factory=lambda: _env_int("SHOWROOM_OPEN_HOUR", 9))
    showroom_close_hour: int = field(default_factory=lambda: _env_int("SHOWROOM_CLOSE_HOUR", 21))

    # Hang up after this long with neither side speaking.
    silence_hangup_seconds: float = field(
        default_factory=lambda: _env_float("SILENCE_HANGUP_SECONDS", 30.0)
    )

    # Bookings and leads are appended to files here, and POSTed as JSON to
    # WEBHOOK_URL when set (a CRM, Zapier, a Google Sheet script...). On LiveKit
    # Cloud the files are lost on restart, so set the webhook there.
    data_dir: str = field(default_factory=lambda: _env("DATA_DIR", "data"))
    webhook_url: str = field(default_factory=lambda: _env("WEBHOOK_URL"))

    def missing(self) -> list[str]:
        """Settings a real call cannot work without."""
        required = {
            "DEEPGRAM_API_KEY": self.deepgram_api_key,
            "SPEECHMATICS_API_KEY": self.speechmatics_api_key,
            "ANTHROPIC_API_KEY": self.anthropic_api_key,
        }
        return [name for name, value in required.items() if not value]


settings = Settings()
