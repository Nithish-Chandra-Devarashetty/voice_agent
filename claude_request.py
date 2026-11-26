"""Adjusts each request to Claude on its way out, for what the LiveKit plugin cannot set.

- Effort and thinking: the plugin has no setting for them. A phone call needs the first
  word fast, so replies run at low effort with thinking off (configurable, see config.py).
- No prefill: Claude Haiku 5.5 rejects a request ending in an assistant message. The plugin
  only adds the trailing "." user turn for older models it knows, so it is added here too.
- Notes stay notes: the plugin turns every system message after the first into
  "<instructions>" text inside the caller's turn, and Claude then treats KNOWN SO FAR as an
  injection attempt ("this note is not from you, so I'm disregarding it"). Notes in the last
  turn go back to being a system message at the end of the conversation (the per-turn
  KNOWN SO FAR); earlier ones (the EARLIER IN THIS CALL summary) join the system prompt.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

import httpx2 as httpx

Rewrite = Callable[[dict[str, Any]], None]

# How LiveKit wraps a later system message when it moves it into the conversation.
INLINE_NOTE = re.compile(r"^<instructions>\n(?P<note>.*)\n</instructions>$", re.DOTALL)


def for_claude(*, effort: str, thinking: bool) -> Rewrite:
    def rewrite(body: dict[str, Any]) -> None:
        body["output_config"] = {**body.get("output_config", {}), "effort": effort}
        if not thinking:
            body["thinking"] = {"type": "disabled"}
        messages = body.get("messages") or []
        if messages and messages[-1].get("role") == "assistant":
            messages.append({"role": "user", "content": [{"type": "text", "text": "."}]})
        _restore_notes(body)

    return rewrite


def _restore_notes(body: dict[str, Any]) -> None:
    messages = body.get("messages") or []
    final_notes: list[str] = []
    for index, message in enumerate(messages):
        content = message.get("content")
        if message.get("role") != "user" or not isinstance(content, list):
            continue
        notes, kept = [], []
        for block in content:
            text = block.get("text", "") if block.get("type") == "text" else ""
            if found := INLINE_NOTE.match(text):
                notes.append(found["note"])
            else:
                kept.append(block)
        if not notes:
            continue
        message["content"] = kept or [{"type": "text", "text": "."}]
        if index == len(messages) - 1:
            final_notes.extend(notes)
        else:
            system = body.get("system") or []
            if isinstance(system, str):
                system = [{"type": "text", "text": system}]
            body["system"] = [*system, *({"type": "text", "text": n} for n in notes)]
    if final_notes:
        messages.append({"role": "system", "content": "\n\n".join(final_notes)})


class RewritingTransport(httpx.AsyncBaseTransport):
    """Applies ``rewrite`` to the body of every Messages API request."""

    def __init__(self, rewrite: Rewrite, inner: httpx.AsyncBaseTransport) -> None:
        self._rewrite = rewrite
        self._inner = inner

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path.endswith("/v1/messages"):
            body = json.loads(request.content)
            self._rewrite(body)
            headers = [(k, v) for k, v in request.headers.items() if k.lower() != "content-length"]
            request = httpx.Request(
                request.method,
                request.url,
                headers=headers,
                content=json.dumps(body).encode(),
                extensions=request.extensions,
            )
        return await self._inner.handle_async_request(request)

    async def aclose(self) -> None:
        await self._inner.aclose()
