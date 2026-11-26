"""One log line per Claude request: tokens, cache use, latency, rate limits and errors."""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

import httpx2 as httpx  # the anthropic 1.x SDK only accepts httpx2 clients
from livekit.agents import llm

from claude_request import Rewrite, RewritingTransport

logger = logging.getLogger("dasco")

RATE_LIMIT_HEADERS = {
    "anthropic-ratelimit-requests-remaining": "remaining_requests",
    "anthropic-ratelimit-input-tokens-remaining": "remaining_input_tokens",
    "anthropic-ratelimit-output-tokens-remaining": "remaining_output_tokens",
    "anthropic-ratelimit-input-tokens-reset": "reset_input_tokens",
    "retry-after": "retry_after",
}


# Claude Haiku 5.5, US dollars per token: input, cache read (0.1x), 5-minute cache write
# (1.25x) and output. Prompts over 100K tokens cost double; a call never gets close.
PRICE_PER_TOKEN = {
    "input": 0.10 / 1e6,
    "cache_read": 0.01 / 1e6,
    "cache_write": 0.125 / 1e6,
    "output": 0.50 / 1e6,
}


def estimated_cost(m: Any) -> float:
    """Dollars for one request. prompt_tokens includes cache reads and writes."""
    read = m.prompt_cached_tokens
    write = getattr(m, "cache_creation_tokens", 0)
    plain = max(m.prompt_tokens - read - write, 0)
    return (
        plain * PRICE_PER_TOKEN["input"]
        + read * PRICE_PER_TOKEN["cache_read"]
        + write * PRICE_PER_TOKEN["cache_write"]
        + m.completion_tokens * PRICE_PER_TOKEN["output"]
    )


class UsageLog:
    """Attach to an LLM and its HTTP client; keeps running totals for the call."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.rate_limits: dict[str, str] = {}
        self.totals: Counter[str] = Counter()
        self.cost_usd = 0.0
        self._clients: list[httpx.AsyncClient] = []

    async def on_response(self, response: httpx.Response) -> None:
        """httpx response hook: runs when the headers arrive, before the stream body."""
        limits = {
            name: response.headers[header]
            for header, name in RATE_LIMIT_HEADERS.items()
            if header in response.headers
        }
        if limits:
            self.rate_limits = limits
        if response.status_code >= 400:
            self.totals["errors"] += 1
            logger.warning(
                "llm_http_error",
                extra={"llm": self.label, "status": response.status_code, **limits},
            )

    def on_metrics(self, m: Any) -> None:
        uncached = m.prompt_tokens - m.prompt_cached_tokens
        cache_writes = getattr(m, "cache_creation_tokens", 0)
        cost = estimated_cost(m)
        self.cost_usd += cost
        self.totals.update(
            requests=1,
            prompt_tokens=m.prompt_tokens,
            cached_tokens=m.prompt_cached_tokens,
            cache_write_tokens=cache_writes,
            uncached_prompt_tokens=uncached,
            completion_tokens=m.completion_tokens,
            reasoning_tokens=m.reasoning_tokens,
        )
        logger.info(
            "llm_request",
            extra={
                "llm": self.label,
                "prompt_tokens": m.prompt_tokens,
                "cached_tokens": m.prompt_cached_tokens,
                "cache_write_tokens": cache_writes,
                "uncached_prompt_tokens": uncached,
                "completion_tokens": m.completion_tokens,
                "reasoning_tokens": m.reasoning_tokens,
                "ttft_s": round(m.ttft, 3),
                "duration_s": round(m.duration, 3),
                "cancelled": m.cancelled,
                "est_cost_usd": round(cost, 6),
                **self.rate_limits,
            },
        )

    def on_error(self, ev: Any) -> None:
        self.totals["errors"] += 1
        logger.error(
            "llm_error",
            extra={
                "llm": self.label,
                "error": str(getattr(ev, "error", ev))[:300],
                "recoverable": getattr(ev, "recoverable", None),
            },
        )

    def attach(self, model: llm.LLM) -> None:
        model.on("metrics_collected", self.on_metrics)
        model.on("error", self.on_error)

    def http_client(self, timeout: httpx.Timeout, rewrite: Rewrite) -> httpx.AsyncClient:
        # Same settings as the LiveKit OpenAI plugin's own client, plus the header hook and
        # the request rewrite (claude_request).
        inner = httpx.AsyncHTTPTransport(
            limits=httpx.Limits(
                max_connections=50, max_keepalive_connections=50, keepalive_expiry=120
            ),
        )
        client = httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            transport=RewritingTransport(rewrite, inner),
            event_hooks={"response": [self.on_response]},
        )
        self._clients.append(client)
        return client

    async def aclose(self) -> None:
        """Log the call's totals and close the HTTP clients (the plugin leaves them open)."""
        logger.info(
            "llm_call_totals",
            extra={"llm": self.label, "est_cost_usd": round(self.cost_usd, 5), **self.totals},
        )
        for client in self._clients:
            await client.aclose()
