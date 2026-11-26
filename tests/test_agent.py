"""The PASCO phone agent, without a live call."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import types
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from livekit import rtc

import agent
import cars
import claude_request
import history
import prompts
import tools
from config import settings
from livekit.agents import llm
from tools import IST, CallData
from speakers import OTHER_VOICE, SpeakerTracker
from usage import UsageLog

NOW = datetime(2026, 10, 13, 11, 0, tzinfo=IST)  # a Tuesday morning


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Records go to a temp folder, no webhook, and the clock is fixed."""
    patched = dataclasses.replace(settings, data_dir=str(tmp_path), webhook_url="")
    monkeypatch.setattr(tools, "settings", patched)
    monkeypatch.setattr(tools, "now", lambda: NOW)


def new_call(caller_phone: str | None = "9876543210", **known: Any) -> Any:
    return types.SimpleNamespace(userdata=CallData(caller_phone=caller_phone, **known))


async def run(tool: Any, ctx: Any = None, **kwargs: Any) -> Any:
    return await getattr(tool, "__wrapped__", tool)(ctx or new_call(), **kwargs)


def records(kind: str) -> list[dict[str, Any]]:
    path = Path(tools.settings.data_dir) / f"{kind}s.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# -- cars ------------------------------------------------------------------------


@pytest.mark.parametrize("said", ["Baleno", "balino", "Maruti Suzuki Baleno", "baleno alpha"])
def test_finds_a_car_however_it_was_said(said: str) -> None:
    assert cars.find_car(said)["model"] == "Baleno"


def test_unknown_car_is_not_guessed() -> None:
    assert cars.find_car("Tesla Model 3") is None
    assert cars.find_car("") is None


def test_details_carry_price_and_offer_types_but_no_offer_amounts() -> None:
    facts = cars.details(cars.find_car("Fronx"))
    assert "lakh" in facts["price"]
    assert facts["offer_types"] and all(isinstance(t, str) for t in facts["offer_types"])


async def test_recommendations_come_from_nexa_by_default() -> None:
    nexa = await run(tools.list_cars)
    assert nexa["cars"] and all(c["brand"] == "NEXA" for c in nexa["cars"])
    # Ignis is known but never suggested.
    assert "Ignis" not in [c["model"] for c in nexa["cars"]]


async def test_first_car_asked_about_is_the_interest_later_ones_are_comparisons() -> None:
    ctx = new_call()
    await run(tools.get_car_details, ctx, car="Fronx")
    await run(tools.get_car_details, ctx, car="Swift")
    assert ctx.userdata.car == "Fronx"


async def test_a_competitor_car_has_no_facts() -> None:
    result = await run(tools.get_car_details, car="Hyundai Creta")
    assert result["found"] is False and "Baleno" in result["known_models"]


# -- name, exchange and the per-turn context ---------------------------------------


async def test_name_and_exchange_show_up_in_known_so_far() -> None:
    ctx = new_call()
    assert "name=unknown" in ctx.userdata.known_so_far()
    await run(tools.save_customer_name, ctx, name="Ravi")
    await run(tools.save_exchange_details, ctx, has_old_car=True, old_car="2015 Swift")
    known = ctx.userdata.known_so_far()
    assert "name=Ravi" in known
    assert "old_car_exchange=yes (2015 Swift)" in known
    assert "phone=known from caller ID" in known
    assert "9876543210" not in known  # never read the number out


# -- test drives -----------------------------------------------------------------


async def test_books_with_the_saved_name_and_the_callers_number() -> None:
    ctx = new_call(name="Ravi", has_old_car=True, old_car="2015 Swift")
    result = await run(tools.book_test_drive, ctx, car="balino", date="2026-10-14", time="17:30")
    assert result["success"] is True
    assert result["when"] == "Wednesday 14 October at 05:30 PM"
    [saved] = records("test_drive")
    assert (saved["name"], saved["phone"], saved["car"]) == ("Ravi", "9876543210", "Baleno")
    assert saved["old_car_exchange"] == "2015 Swift"
    assert "Baleno on Wednesday" in ctx.userdata.known_so_far()


async def test_a_different_number_given_on_the_call_wins() -> None:
    await run(
        tools.book_test_drive,
        new_call(name="Ravi"),
        car="Fronx",
        date="2026-10-14",
        time="10:00",
        phone="+91 91234 56789",
    )
    assert records("test_drive")[0]["phone"] == "9123456789"


@pytest.mark.parametrize(
    ("date", "time", "problem"),
    [
        ("2026-10-12", "10:00", "passed"),
        ("2026-10-14", "22:00", "closed"),
        ("2026-10-14", "07:30", "closed"),
        ("tomorrow", "5 pm", "YYYY-MM-DD"),
    ],
)
async def test_bad_times_are_refused(date: str, time: str, problem: str) -> None:
    result = await run(
        tools.book_test_drive, new_call(name="Ravi"), car="Baleno", date=date, time=time
    )
    assert result["success"] is False and problem in result["problem"]


async def test_arena_test_drives_are_routed_to_a_callback() -> None:
    result = await run(
        tools.book_test_drive, new_call(name="Ravi"), car="Swift", date="2026-10-14", time="10:00"
    )
    assert result["success"] is False and "save_lead" in result["problem"]


async def test_booking_needs_a_name_and_a_number() -> None:
    no_name = await run(tools.book_test_drive, car="Baleno", date="2026-10-14", time="10:00")
    assert no_name["success"] is False and "name" in no_name["problem"]
    no_phone = await run(
        tools.book_test_drive,
        new_call(caller_phone=None, name="Ravi"),
        car="Baleno",
        date="2026-10-14",
        time="10:00",
    )
    assert no_phone["success"] is False and "phone" in no_phone["problem"]


# -- leads and complaints --------------------------------------------------------


@pytest.mark.parametrize(("interest", "team"), [("Grand Vitara", "nexa"), ("Swift", "arena")])
async def test_leads_go_to_the_right_team(interest: str, team: str) -> None:
    ctx = new_call(name="Asha")
    result = await run(tools.save_lead, ctx, interest=interest, notes="budget 15L")
    assert result["success"] is True and result["team"] == team
    assert records("lead")[0]["team"] == team and records("lead")[0]["name"] == "Asha"
    assert "callback_saved=yes" in ctx.userdata.known_so_far()


@pytest.mark.parametrize(("said", "saved"), [("known", "9876543210"), ("98765", None)])
async def test_a_phone_placeholder_means_caller_id_but_a_bad_number_is_refused(
    said: str, saved: str | None
) -> None:
    # A model once sent phone="known", echoing KNOWN SO FAR.
    result = await run(tools.record_complaint, new_call(name="Asha"), details="Rude", phone=said)
    assert result["success"] is (saved is not None)
    if saved:
        assert records("complaint")[0]["phone"] == saved


async def test_complaints_go_to_support() -> None:
    result = await run(tools.record_complaint, new_call(name="Asha"), details="Rude staff")
    assert result["success"] is True
    assert records("complaint")[0]["team"] == "support"


async def test_webhook_failure_still_saves_locally(monkeypatch: pytest.MonkeyPatch) -> None:
    broken = dataclasses.replace(tools.settings, webhook_url="http://127.0.0.1:9/nothing")
    monkeypatch.setattr(tools, "settings", broken)
    result = await run(tools.save_lead, interest="Jimny")
    assert result["success"] is True and len(records("lead")) == 1


# -- phone numbers and the prompt ------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("+919876543210", "9876543210"), ("09876543210", "9876543210"), ("12345", None)],
)
def test_normalizes_indian_numbers(raw: str, expected: str | None) -> None:
    assert tools.normalize_phone(raw) == expected


def _participant(kind: int, attributes: dict[str, str]) -> Any:
    return types.SimpleNamespace(kind=kind, attributes=attributes)


def test_reads_the_caller_number_from_a_phone_call() -> None:
    sip = rtc.ParticipantKind.PARTICIPANT_KIND_SIP
    standard = rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD
    phone_call = _participant(sip, {"sip.phoneNumber": "+919876543210"})
    assert agent.caller_number(phone_call) == "9876543210"
    assert agent.caller_number(_participant(standard, {"sip.phoneNumber": "+91987"})) is None


def test_greeting_asks_for_the_name_first() -> None:
    assert "your name" in prompts.greeting()


def test_prompt_carries_v1_rules_in_english_only() -> None:
    text = prompts.instructions()
    for rule in ("SALES FLOW", "ARENA CARS", "COMPETITOR CARS", "COMPLAINTS", "OFF-TOPIC"):
        assert f"# {rule}" in text
    assert "Speak English only" in text
    assert "Hinglish" not in text
    assert "Baleno" in text and "Swift" in text  # NEXA and Arena ranges listed


def test_registers_under_its_dispatch_name() -> None:
    assert agent.server._agent_name == settings.agent_name


def test_llm_is_claude_haiku_with_caching(monkeypatch: pytest.MonkeyPatch) -> None:
    patched = dataclasses.replace(settings, anthropic_api_key="test-key")
    monkeypatch.setattr(agent, "settings", patched)
    llm = agent.build_llm()
    assert llm.model == "claude-haiku-5-5"
    assert llm._opts.caching == "ephemeral"
    assert "ANTHROPIC_API_KEY" in dataclasses.replace(settings, anthropic_api_key="").missing()


async def test_null_optional_fields_are_accepted() -> None:
    # Models send null for optional fields they are not using.
    ctx = new_call(name="Ravi")
    booked = await run(
        tools.book_test_drive,
        ctx,
        car="Fronx",
        date="2026-10-14",
        time="17:00",
        name=None,
        phone=None,
    )
    assert booked["success"] is True and records("test_drive")[0]["phone"] == "9876543210"
    lead = await run(tools.save_lead, ctx, interest="Swift", notes=None, name=None, phone=None)
    assert lead["success"] is True
    assert (await run(tools.save_exchange_details, ctx, has_old_car=False, old_car=None))["saved"]
    assert (await run(tools.list_cars, ctx, brand=None))["cars"]


# -- token use: a cacheable prefix, compacted history, request logging ------------


def test_prompt_is_identical_on_every_call_so_claude_can_cache_it() -> None:
    # The clock lives in KNOWN SO FAR, not in the cached prefix.
    assert prompts.instructions() == prompts.instructions()
    assert "2026" not in prompts.instructions()
    known = new_call().userdata.known_so_far()
    assert "now=Tuesday 13 October 2026, 11:00 AM" in known


async def test_complaint_shows_up_in_known_so_far() -> None:
    ctx = new_call(name="Asha")
    await run(tools.record_complaint, ctx, details="Rude staff")
    assert "complaint_recorded=yes" in ctx.userdata.known_so_far()


def _conversation(turns: int) -> llm.ChatContext:
    chat = llm.ChatContext()
    chat.add_message(role="system", content="INSTRUCTIONS")
    chat.add_message(role="assistant", content="Hello, may I know your name?")
    for n in range(turns):
        chat.add_message(role="user", content=f"question {n}")
        if n == 1:
            name = "get_car_details"
            chat.insert(llm.FunctionCall(call_id="c1", name=name, arguments='{"car":"XL6"}'))
            chat.insert(
                llm.FunctionCallOutput(
                    call_id="c1", name=name, output="{'price': 'x'}", is_error=False
                )
            )
        chat.add_message(role="assistant", content=f"answer {n}", interrupted=n == 2)
    return chat


async def _summarise(text: str) -> str:
    return f"SUMMARY OF {text.count('Customer:')} TURNS"


async def test_short_calls_are_left_alone() -> None:
    chat = _conversation(history.COMPACT_AFTER_TURNS)
    assert await history.compact(chat, _summarise) is None


async def test_older_turns_are_summarised_and_recent_ones_kept_verbatim() -> None:
    seen: list[str] = []

    async def summarise(text: str) -> str:
        seen.append(text)
        return "Asked about the XL6."

    chat = _conversation(history.COMPACT_AFTER_TURNS + 1)
    result = await history.compact(chat, summarise)
    assert result is not None
    # The summariser saw the tool lookup and knew a reply had been cut off.
    assert "get_car_details" in seen[0] and "(cut off by the customer)" in seen[0]
    compacted = result.apply(chat)
    texts = [i.text_content for i in compacted.items if isinstance(i, llm.ChatMessage)]
    assert texts[0] == "INSTRUCTIONS"
    assert texts[1] == f"{history.SUMMARY_TAG}: Asked about the XL6."
    users = [i for i in compacted.items if isinstance(i, llm.ChatMessage) and i.role == "user"]
    assert len(users) == history.KEEP_TURNS
    assert users[-1].text_content == f"question {history.COMPACT_AFTER_TURNS}"
    assert not any(i.type.startswith("function_call") for i in compacted.items)


async def test_turns_added_while_summarising_are_kept() -> None:
    chat = _conversation(history.COMPACT_AFTER_TURNS + 1)
    result = await history.compact(chat, _summarise)
    chat.add_message(role="user", content="said during the summary")
    compacted = result.apply(chat)
    assert compacted.items[-1].text_content == "said during the summary"


async def test_a_second_summary_folds_in_the_first() -> None:
    chat = _conversation(history.COMPACT_AFTER_TURNS + 1)
    chat = (await history.compact(chat, _summarise)).apply(chat)
    for n in range(10):
        chat.add_message(role="user", content=f"later {n}")
        chat.add_message(role="assistant", content=f"reply {n}")
    seen: list[str] = []

    async def summarise(text: str) -> str:
        seen.append(text)
        return "second"

    chat = (await history.compact(chat, summarise)).apply(chat)
    assert seen[0].startswith("Earlier summary: ")
    notes = [i for i in chat.items if isinstance(i, llm.ChatMessage) and i.extra.get("is_summary")]
    assert [n.text_content for n in notes] == [f"{history.SUMMARY_TAG}: second"]


async def test_a_failed_summary_keeps_the_full_history() -> None:
    async def broken(text: str) -> str:
        raise RuntimeError("429")

    chat = _conversation(history.COMPACT_AFTER_TURNS + 1)
    assert await history.compact(chat, broken) is None


async def test_usage_log_records_tokens_cache_use_cost_and_rate_limits(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import httpx2

    usage = UsageLog("reply")
    await usage.on_response(
        httpx2.Response(200, headers={"anthropic-ratelimit-input-tokens-remaining": "7000"})
    )
    metrics = types.SimpleNamespace(
        prompt_tokens=2000, prompt_cached_tokens=1536, cache_creation_tokens=0,
        completion_tokens=40, reasoning_tokens=0, ttft=0.3, duration=0.5, cancelled=False,
    )
    with caplog.at_level("INFO", logger="dasco"):
        usage.on_metrics(metrics)
    [record] = [r for r in caplog.records if r.msg == "llm_request"]
    assert record.cached_tokens == 1536 and record.uncached_prompt_tokens == 464
    assert record.remaining_input_tokens == "7000"
    assert record.est_cost_usd > 0 and usage.totals["prompt_tokens"] == 2000


# -- speaker diarization: other voices are marked, never muted -------------------


def _said(text: str, speaker: str | None, final: bool = True) -> Any:
    return types.SimpleNamespace(transcript=text, speaker_id=speaker, is_final=final)


def test_the_first_voice_to_answer_is_the_caller_and_others_are_marked() -> None:
    voices = SpeakerTracker()
    voices.on_transcript(_said("Rav", "S1", final=False))  # interim: not counted
    voices.on_transcript(_said("Ravi here", "S1"))
    assert voices.caller == "S1" and voices.turn_is_other_voice() is False
    voices.on_transcript(_said("Ask about the mileage", "S2"))
    assert voices.turn_is_other_voice() is True
    # The caller and someone else in one turn: it is the caller's turn.
    voices.on_transcript(_said("What about", "S2"))
    voices.on_transcript(_said("the price?", "S1"))
    assert voices.turn_is_other_voice() is False


def test_an_unclear_voice_counts_as_the_caller() -> None:
    voices = SpeakerTracker()
    voices.on_transcript(_said("hmm", "UU"))  # before anyone is known: not the caller yet
    assert voices.caller is None and voices.turn_is_other_voice() is False
    voices.on_transcript(_said("Ravi here", "S1"))
    voices.turn_is_other_voice()
    voices.on_transcript(_said("Saturday at six", "UU"))
    assert voices.turn_is_other_voice() is False


def test_a_short_reply_is_the_caller_whatever_its_label() -> None:
    # A real call: the caller's goodbye came through as S2 and went unanswered.
    voices = SpeakerTracker()
    voices.on_transcript(_said("My name is Nitish", "S1"))
    voices.turn_is_other_voice()
    voices.on_transcript(_said("No.", "S2"))
    assert voices.turn_is_other_voice() is False
    voices.on_transcript(_said("What is this call about?", "S3"))
    assert voices.turn_is_other_voice() is True


def test_a_voice_that_keeps_talking_alone_becomes_the_caller() -> None:
    voices = SpeakerTracker()
    voices.on_transcript(_said("My name is Nitish", "S1"))
    voices.turn_is_other_voice()
    voices.on_transcript(_said("I want the petrol variant please", "S2"))
    assert voices.turn_is_other_voice() is True
    voices.on_transcript(_said("And what is the mileage of it", "S2"))
    assert voices.turn_is_other_voice() is False and voices.caller == "S2"
    voices.on_transcript(_said("Tell me about the Baleno please", "S1"))
    assert voices.turn_is_other_voice() is True  # the old label is now the other voice


def test_a_long_reply_stops_at_a_sentence_end() -> None:
    long = " ".join(["word"] * agent.MAX_REPLY_WORDS) + ". Then it keeps going"
    assert agent.run_on_cut("", long) == " ".join(["word"] * agent.MAX_REPLY_WORDS) + "."
    short = "The Grand Vitara is a family SUV. Would you like a test drive?"
    assert agent.run_on_cut("", short) is None


def test_the_prompt_ends_the_call_on_no_after_anything_else() -> None:
    text = prompts.instructions()
    assert "Never say goodbye without calling end_call" in text
    assert "35 words" in text


@pytest.mark.parametrize(
    ("opening", "decision"),
    [
        ("IGNORE", "ignore"),
        ("  IGNORE", "ignore"),
        ("IGN", "hold"),
        ("", "hold"),
        ("I", "hold"),
        ("I have booked", "speak"),
        ("Ignore that, the Baleno", "speak"),  # an ordinary sentence starting with "Ignore"
        ("The Baleno", "speak"),
    ],
)
def test_a_reply_of_ignore_is_never_spoken(opening: str, decision: str) -> None:
    assert agent.ignore_decision(opening) == decision


def test_the_prompt_explains_other_voices() -> None:
    text = prompts.instructions()
    assert OTHER_VOICE in text and "exactly IGNORE" in text


def test_speech_to_text_is_speechmatics_with_diarization(monkeypatch: pytest.MonkeyPatch) -> None:
    from livekit.plugins import speechmatics

    patched = dataclasses.replace(settings, speechmatics_api_key="test-key")
    monkeypatch.setattr(agent, "settings", patched)
    stt = agent.build_stt(vad=None)
    assert isinstance(stt, speechmatics.STT)
    assert stt._stt_options.enable_diarization is True
    assert "SPEECHMATICS_API_KEY" in dataclasses.replace(
        settings, speechmatics_api_key=""
    ).missing()


# -- a reply that runs on past its turn is cut ------------------------------------


@pytest.mark.parametrize(
    ("said", "text", "kept"),
    [
        ("", "Which time works for you?User", "Which time works for you?"),
        ("Which time works for you?", "User", ""),
        ("Which time works for you?", "\n\nAssistant: Apologies", ""),
        ("", "Is there anything else?", None),
        ("", "A user manual comes with the car.", None),  # lowercase, mid-sentence
        # Five rephrasings glued together (a real call): only the first question is spoken.
        (
            "",
            "Which model would you like, for example the Baleno?Could you tell me which",
            "Which model would you like, for example the Baleno?",
        ),
        ("Which model would you like?", " Do you have", ""),
        ("", "Is it the Baleno? Great choice.", "Is it the Baleno?"),
        ("", "Would you like a test drive, Ravi?", None),
        ("Would you like a test drive?", " ", None),  # only a space so far: keep waiting
    ],
)
def test_a_reply_running_into_the_next_turn_is_cut(
    said: str, text: str, kept: str | None
) -> None:
    assert agent.run_on_cut(said, text) == kept


# -- every Claude request gets fast settings and a valid ending -----------------


def _body(last_role: str = "user") -> dict[str, Any]:
    return {"model": "claude-haiku-5-5", "messages": [{"role": last_role, "content": "hi"}]}


def test_requests_run_at_low_effort_with_thinking_off() -> None:
    body = _body()
    claude_request.for_claude(effort="low", thinking=False)(body)
    assert body["output_config"] == {"effort": "low"}
    assert body["thinking"] == {"type": "disabled"}


def test_thinking_can_be_left_on() -> None:
    body = _body()
    claude_request.for_claude(effort="medium", thinking=True)(body)
    assert "thinking" not in body and body["output_config"]["effort"] == "medium"


def test_a_request_never_ends_on_an_assistant_message() -> None:
    # Claude Haiku 5.5 does not accept a prefilled reply.
    body = _body(last_role="assistant")
    claude_request.for_claude(effort="low", thinking=False)(body)
    assert body["messages"][-1]["role"] == "user"
    untouched = _body()
    claude_request.for_claude(effort="low", thinking=False)(untouched)
    assert len(untouched["messages"]) == 1


async def test_the_rewrite_is_applied_to_outgoing_requests() -> None:
    import httpx2

    sent: list[dict[str, Any]] = []

    def capture(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content))
        return httpx2.Response(200, json={})

    rewrite = claude_request.for_claude(effort="low", thinking=False)
    client = UsageLog("reply").http_client(httpx2.Timeout(5), rewrite)
    client._transport._inner = httpx2.MockTransport(capture)
    await client.post("https://api.anthropic.com/v1/messages", json=_body())
    assert sent[0]["output_config"] == {"effort": "low"}


def test_cost_estimate_prices_cache_reads_and_writes() -> None:
    from usage import estimated_cost

    m = types.SimpleNamespace(
        prompt_tokens=3000, prompt_cached_tokens=2000, cache_creation_tokens=500,
        completion_tokens=100,
    )
    # 500 plain input + 2000 cache reads + 500 cache writes + 100 output tokens.
    expected = 500 * 0.10e-6 + 2000 * 0.01e-6 + 500 * 0.125e-6 + 100 * 0.50e-6
    assert abs(estimated_cost(m) - expected) < 1e-12


def test_call_notes_reach_claude_as_system_messages_not_as_the_callers_words() -> None:
    # LiveKit wraps later system messages as <instructions> text in the caller's turn;
    # Claude then reads KNOWN SO FAR as an injection attempt.
    from livekit.agents.llm._provider_format import anthropic as provider_format

    chat = llm.ChatContext()
    chat.add_message(role="system", content="INSTRUCTIONS")
    chat.add_message(role="system", content="EARLIER IN THIS CALL: asked about the Baleno")
    chat.add_message(role="user", content="hi")
    chat.add_message(role="assistant", content="hello")
    chat.add_message(role="user", content="price?")
    chat.add_message(role="system", content="KNOWN SO FAR: name=Ravi")
    messages, extra = provider_format.to_chat_ctx(chat)
    body = {"system": [{"type": "text", "text": t} for t in extra.system_messages],
            "messages": messages}
    claude_request.for_claude(effort="low", thinking=False)(body)
    assert [b["text"] for b in body["system"]] == [
        "INSTRUCTIONS", "EARLIER IN THIS CALL: asked about the Baleno"
    ]
    assert body["messages"][-1] == {"role": "system", "content": "KNOWN SO FAR: name=Ravi"}
    assert "instructions>" not in json.dumps(body["messages"])
