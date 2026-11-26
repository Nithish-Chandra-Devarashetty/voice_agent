"""Keeps the conversation sent to the LLM short: recent turns verbatim, older ones summarised.

Without this every request carries the whole call, including every car-details lookup.
Compaction runs only once the history has grown past COMPACT_AFTER_TURNS, and then cuts it
back to KEEP_TURNS, so between compactions the history is append-only and Claude's prompt
cache keeps hitting on it. Facts the call depends on (name, car, booking...) never rely on
the summary: they are restated every turn in the KNOWN SO FAR line.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from livekit.agents import llm

logger = logging.getLogger("dasco")

# Customer turns kept word for word. Older turns are folded into one summary.
KEEP_TURNS = 3
COMPACT_AFTER_TURNS = 6
SUMMARY_TAG = "EARLIER IN THIS CALL"

SUMMARY_PROMPT = """You compress the older part of a phone call between a car showroom's \
sales assistant and a customer into a short note the assistant will read.
Write at most 70 words of plain text: which cars were discussed and what the customer \
asked, what the assistant already told them (prices, mileage, features, offers), \
anything the customer wants that is still pending, and what the car conversation was \
about before any detour. Leave out greetings and filler, and do not mention tools or \
reference numbers. Never add facts that are not in the conversation."""

Summarizer = Callable[[str], Awaitable[str]]


def _is_summary(item: llm.ChatItem) -> bool:
    return isinstance(item, llm.ChatMessage) and bool(item.extra.get("is_summary"))


def _is_turn(item: llm.ChatItem) -> bool:
    return isinstance(item, llm.ChatMessage) and item.role == "user" and not _is_summary(item)


def _render(item: llm.ChatItem) -> str | None:
    if isinstance(item, llm.ChatMessage):
        if item.role not in ("user", "assistant") or _is_summary(item):
            return None
        text = (item.text_content or "").strip()
        if not text:
            return None
        who = "Customer" if item.role == "user" else "Assistant"
        return f"{who}: {text}{' (cut off by the customer)' if item.interrupted else ''}"
    if isinstance(item, llm.FunctionCall):
        return f"[{item.name} {item.arguments}]"
    if isinstance(item, llm.FunctionCallOutput):
        return f"[result: {item.output[:600]}]"
    return None


def split_point(items: list[llm.ChatItem]) -> int | None:
    """Index of the first item to keep verbatim, or None if no compaction is due."""
    turn_starts = [i for i, item in enumerate(items) if _is_turn(item)]
    if len(turn_starts) <= COMPACT_AFTER_TURNS:
        return None
    return turn_starts[-KEEP_TURNS]


@dataclass
class Compaction:
    """Older items to drop and the note that replaces them."""

    dropped: set[str]
    note: llm.ChatMessage

    def apply(self, current: llm.ChatContext) -> llm.ChatContext:
        """``current`` without the dropped items, the note after the instructions.

        Applied to the context as it is when the summary is ready, so turns added while it
        was being written are kept.
        """
        items = [i for i in current.items if i.id not in self.dropped]
        at = next(
            (n for n, i in enumerate(items) if not _is_instructions(i)),
            len(items),
        )
        return llm.ChatContext([*items[:at], self.note, *items[at:]])


def _is_instructions(item: llm.ChatItem) -> bool:
    return (
        isinstance(item, llm.ChatMessage)
        and item.role in ("system", "developer")
        and not _is_summary(item)
    )


async def compact(chat_ctx: llm.ChatContext, summarize: Summarizer) -> Compaction | None:
    """What to fold into a summary, or None while the history is still short."""
    items = list(chat_ctx.items)
    split = split_point(items)
    if split is None:
        return None
    head = [i for i in items[:split] if not _is_instructions(i)]
    previous = next((i for i in head if _is_summary(i)), None)
    lines = [line for item in head if (line := _render(item))]
    if not lines:
        return None
    source = "\n".join(lines)
    if previous is not None:
        source = f"Earlier summary: {previous.text_content}\n{source}"

    try:
        summary = (await summarize(source)).strip()
    except Exception as exc:  # noqa: BLE001 - a summary is an optimisation, never fatal
        logger.warning("history_summary_failed", extra={"error": str(exc)[:200]})
        return None  # keep the full history this time
    if not summary:
        return None
    note = llm.ChatMessage(
        role="system", content=[f"{SUMMARY_TAG}: {summary}"], extra={"is_summary": True}
    )
    return Compaction(dropped={i.id for i in head}, note=note)
