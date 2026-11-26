"""PASCO Motors phone agent: Speechmatics in, Claude Haiku 5.5, Deepgram voice, English only.

Run locally:   python agent.py dev        (or: python agent.py console)
In the cloud:  python agent.py start      (what the Dockerfile runs)
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
from typing import Any

import anthropic
import httpx2 as httpx  # the anthropic 1.x SDK only accepts httpx2 clients
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    EndpointingOptions,
    InterruptionOptions,
    JobContext,
    JobProcess,
    TurnHandlingOptions,
    cli,
    room_io,
)
from livekit.agents import llm as agents_llm
from livekit.agents.beta.tools import EndCallTool
from livekit.agents.voice.room_io.types import NoiseCancellationParams
from livekit.plugins import anthropic as claude
from livekit.plugins import deepgram, noise_cancellation, silero, speechmatics
from livekit.plugins.turn_detector.english import EnglishModel

import cars
import claude_request
import history
import prompts
from config import settings
from speakers import OTHER_VOICE, SpeakerTracker
from tools import ALL_TOOLS, CallData, mask_phone, normalize_phone
from usage import UsageLog

logger = logging.getLogger("dasco")

# Set by LiveKit on a phone caller: the number they are calling from.
CALLER_NUMBER_ATTRIBUTE = "sip.phoneNumber"

# A model can run past the end of its turn, write the next speaker's label and answer
# again: "...works for you?User Apologies, could you...". Speech stops there.
ROLE_LABEL_AFTER_SENTENCE = re.compile(r"[.!?]\s*(User|Assistant)\b")
# A reply ends with at most one question (prompt rule). Anything after it is the model
# running on, e.g. five rephrasings of the same question glued together.
QUESTION_END = re.compile(r"\?[\"'’)]*")
# Phone replies stay short (prompt: SPEAKING STYLE). If the model still runs long, speech
# stops at the first sentence end past this many words (about twenty seconds spoken).
MAX_REPLY_WORDS = 50
SENTENCE_END = re.compile(r"[.!?][\"'’)]*(?=\s)")

# The LiveKit Anthropic plugin's default timeouts: up to 30 s between streamed chunks.
LLM_TIMEOUT = httpx.Timeout(5.0, read=30.0)


# The whole reply when another voice's remark is not for the agent (prompt: OTHER VOICES).
IGNORE_REPLY = "IGNORE"


class DascoAgent(Agent):
    def __init__(
        self,
        summary_llm: agents_llm.LLM | None = None,
        speakers: SpeakerTracker | None = None,
    ) -> None:
        end_call = EndCallTool(
            extra_description="Only after any booking or callback they asked for is saved.",
            end_instructions="Say a one-sentence goodbye.",
            # Hidden during the greeting, so the call cannot end before it starts.
            ignore_on_enter=True,
            on_tool_completed=self._skip_second_goodbye,
        )
        # Instructions and tools never change during a call, so Claude caches them.
        super().__init__(instructions=prompts.instructions(), tools=[*ALL_TOOLS, end_call])
        # Whether the reply being generated has said anything yet.
        self._reply_has_text = False
        self._speakers = speakers
        self.summary_usage = UsageLog("summary")
        self._summary_llm = summary_llm or build_llm(self.summary_usage, max_tokens=400)
        self._compacting: asyncio.Task[None] | None = None
        # Other-voice remarks the agent chose not to answer: kept out of the history, so the
        # model never answers them later.
        self._ignored: set[str] = set()

    async def on_enter(self) -> None:
        self.session.on("conversation_item_added", self._maybe_compact)
        self.session.on("conversation_item_added", self._forget_if_ignored)
        # Spoken straight away, without waiting for the LLM: no dead air.
        self.session.say(prompts.greeting(), allow_interruptions=True)

    def _maybe_compact(self, ev: Any) -> None:
        # After each reply, in the background: never delays what the caller hears.
        item = ev.item
        if not (isinstance(item, agents_llm.ChatMessage) and item.role == "assistant"):
            return
        if self._compacting is None and history.split_point(self.chat_ctx.items) is not None:
            self._compacting = asyncio.create_task(self._compact())

    def _forget_if_ignored(self, ev: Any) -> None:
        # LiveKit may add the remark to the history after the reply was already ignored.
        if ev.item.id in self._ignored:
            asyncio.create_task(self._forget_ignored())

    async def _forget_ignored(self) -> None:
        if any(item.id in self._ignored for item in self.chat_ctx.items):
            ctx = self.chat_ctx.copy()
            ctx.items[:] = [i for i in ctx.items if i.id not in self._ignored]
            await self.update_chat_ctx(ctx)

    async def _compact(self) -> None:
        try:
            result = await history.compact(self.chat_ctx, self._summarize)
            if result is not None:
                await self.update_chat_ctx(result.apply(self.chat_ctx))
                logger.info("history_compacted", extra={"items_dropped": len(result.dropped)})
        except Exception:
            logger.exception("history_compaction_failed")
        finally:
            self._compacting = None

    async def _summarize(self, text: str) -> str:
        ctx = agents_llm.ChatContext()
        ctx.add_message(role="system", content=history.SUMMARY_PROMPT)
        ctx.add_message(role="user", content=text)
        parts: list[str] = []
        async with self._summary_llm.chat(chat_ctx=ctx) as stream:
            async for chunk in stream:
                if chunk.delta and chunk.delta.content:
                    parts.append(chunk.delta.content)
        return "".join(parts)

    async def on_user_turn_completed(
        self, turn_ctx: agents_llm.ChatContext, new_message: agents_llm.ChatMessage
    ) -> None:
        if self._speakers is not None and self._speakers.turn_is_other_voice():
            logger.info("other_voice_turn")
            new_message.content = [f"{OTHER_VOICE} {new_message.text_content or ''}"]

    async def llm_node(
        self, chat_ctx: agents_llm.ChatContext, tools: list[agents_llm.Tool], model_settings: Any
    ) -> Any:
        self._reply_has_text = False
        # Restated every turn, so the model never re-asks for what it already has.
        chat_ctx.add_message(role="system", content=self.session.userdata.known_so_far())
        chat_ctx.items[:] = [i for i in chat_ctx.items if i.id not in self._ignored]
        last_user = next(
            (i for i in reversed(chat_ctx.items) if getattr(i, "role", None) == "user"), None
        )
        said = ""
        held: str | None = ""  # the reply's opening, held until it cannot be IGNORE_REPLY
        # aclosing: a cut reply also stops the Claude stream, so it stops spending tokens.
        async with contextlib.aclosing(
            Agent.default.llm_node(self, chat_ctx, tools, model_settings)
        ) as stream:
            async for chunk in stream:
                text = chunk if isinstance(chunk, str) else getattr(chunk.delta, "content", None)
                if text and held is not None:
                    held += text
                    decision = ignore_decision(held)
                    if decision == "ignore":
                        logger.info("other_voice_ignored")
                        if last_user is not None:
                            self._ignored.add(last_user.id)
                            await self._forget_ignored()
                        return
                    if decision == "hold":
                        continue
                    text, held = held, None
                    if isinstance(chunk, str):
                        chunk = text
                    else:
                        chunk.delta.content = text
                if text:
                    keep = run_on_cut(said, text)
                    if keep is not None:
                        logger.warning("llm_run_on_cut", extra={"dropped": text[len(keep) :]})
                        if keep.strip():
                            self._reply_has_text = True
                            yield keep
                        return
                    said += text
                if not self._reply_has_text:
                    self._reply_has_text = bool(text and text.strip())
                yield chunk
        if held and ignore_decision(held) != "ignore":  # a reply shorter than IGNORE_REPLY
            self._reply_has_text = bool(held.strip())
            yield held

    async def _skip_second_goodbye(self, ev: Any) -> None:
        # The model usually says goodbye and calls end_call in one reply. A tool
        # result would make it say goodbye a second time, so give none then.
        if self._reply_has_text:
            ev.output = None


def ignore_decision(opening: str) -> str:
    """"ignore" if the reply is IGNORE_REPLY, "hold" while it still could be, else "speak"."""
    start = opening.lstrip()
    if start.startswith(IGNORE_REPLY):
        return "ignore"
    return "hold" if IGNORE_REPLY.startswith(start) else "speak"


def run_on_cut(said: str, text: str) -> str | None:
    """The part of ``text`` to keep if the reply runs on past its turn, else None."""
    full = said + text
    cuts = []
    if label := ROLE_LABEL_AFTER_SENTENCE.search(full):
        cuts.append(label.start(1))
    if (question := QUESTION_END.search(full)) and any(c.isalpha() for c in full[question.end() :]):
        cuts.append(question.end())
    for end in SENTENCE_END.finditer(full):
        if len(full[: end.end()].split()) >= MAX_REPLY_WORDS:
            if any(c.isalpha() for c in full[end.end() :]):
                cuts.append(end.end())
            break
    if not cuts:
        return None
    return text[: max(0, min(cuts) - len(said))]


def caller_number(participant: rtc.RemoteParticipant) -> str | None:
    if participant.kind != rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
        return None
    return normalize_phone((participant.attributes or {}).get(CALLER_NUMBER_ATTRIBUTE, ""))


def pick_noise_cancellation(params: NoiseCancellationParams) -> Any:
    """Strip background noise and other voices; phone audio gets the telephony model."""
    if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
        return noise_cancellation.BVCTelephony()
    return noise_cancellation.BVC()


def hang_up_on_silence(session: AgentSession, ctx: JobContext) -> None:
    """End the call once the caller has been silent for SILENCE_HANGUP_SECONDS."""
    ending: list[asyncio.Task[None]] = []

    async def _end() -> None:
        logger.info("caller_silent_hanging_up")
        await session.say(
            f"I haven't heard anything, so I'll end the call now. Thank you for calling "
            f"{settings.dealership_name}.",
            allow_interruptions=False,
        )
        await ctx.delete_room()  # disconnects a phone caller too
        ctx.shutdown(reason="caller_silent")

    @session.on("user_state_changed")
    def _on_user_state(ev: Any) -> None:
        if ev.new_state == "away" and not ending:
            ending.append(asyncio.create_task(_end()))


def build_stt(vad: Any) -> speechmatics.STT:
    """Speechmatics with speaker diarization, so the agent can lock onto the caller."""
    return speechmatics.STT(
        api_key=settings.speechmatics_api_key,
        language=settings.speechmatics_language,
        enable_diarization=True,
        additional_vocab=[speechmatics.AdditionalVocabEntry(content=t) for t in cars.keyterms()],
        # Silero marks where speech ends; LiveKit's turn detector still decides the turn.
        vad=vad,
    )


def build_llm(usage: UsageLog | None = None, *, max_tokens: int | None = None) -> claude.LLM:
    """Claude, logging every request's tokens, cache use, cost, latency and rate limits."""
    usage = usage or UsageLog("reply")
    rewrite = claude_request.for_claude(
        effort=settings.claude_effort, thinking=settings.claude_thinking
    )
    client = anthropic.AsyncAnthropic(
        api_key=settings.anthropic_api_key,
        max_retries=0,  # LiveKit retries
        http_client=usage.http_client(LLM_TIMEOUT, rewrite),
    )
    model_llm = claude.LLM(
        model=settings.claude_model,
        api_key=settings.anthropic_api_key,
        client=client,
        max_tokens=max_tokens or settings.claude_max_tokens,
        # Caches the instructions, tools and call history: later turns read them at a
        # tenth of the input price.
        caching="ephemeral",
    )
    usage.attach(model_llm)
    return model_llm


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


server = AgentServer(setup_fnc=prewarm)


@server.rtc_session(agent_name=settings.agent_name)
async def entrypoint(ctx: JobContext) -> None:
    if missing := settings.missing():
        logger.error("configuration_incomplete", extra={"missing": missing})

    await ctx.connect()
    participant = await ctx.wait_for_participant()
    phone = caller_number(participant)
    logger.info("call_started", extra={"room": ctx.room.name, "caller": mask_phone(phone)})

    reply_usage = UsageLog("reply")
    stt = build_stt(ctx.proc.userdata["vad"])
    session: AgentSession[CallData] = AgentSession(
        userdata=CallData(caller_phone=phone),
        stt=stt,
        llm=build_llm(reply_usage),
        tts=deepgram.TTS(model=settings.tts_model, api_key=settings.deepgram_api_key),
        vad=ctx.proc.userdata["vad"],
        turn_handling=TurnHandlingOptions(
            turn_detection=EnglishModel(),
            # Callers pause mid-sentence (reading out a number); wait a little longer.
            endpointing=EndpointingOptions(min_delay=0.6, max_delay=6.0),
            # A barge-in needs two real words, so "hmm" or "okay" does not cut the agent off.
            interruption=InterruptionOptions(min_words=2),
        ),
        # Two tool calls (save, then end the call) must fit in one turn.
        max_tool_steps=4,
        user_away_timeout=settings.silence_hangup_seconds,
    )
    hang_up_on_silence(session, ctx)
    speakers = SpeakerTracker()
    speakers.attach(session)
    dasco = DascoAgent(speakers=speakers)

    async def _log_usage() -> None:
        await reply_usage.aclose()
        await dasco.summary_usage.aclose()

    ctx.add_shutdown_callback(_log_usage)

    await session.start(
        agent=dasco,
        room=ctx.room,
        room_options=room_io.RoomOptions(
            participant_identity=participant.identity,
            audio_input=room_io.AudioInputOptions(noise_cancellation=pick_noise_cancellation),
        ),
    )


if __name__ == "__main__":
    cli.run_app(server)
