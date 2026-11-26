"""Speaker diarization: marks turns that came from someone other than the caller.

Speechmatics labels each voice (S1, S2...). The first labelled voice to answer the greeting
is taken as the caller. Every voice still reaches the agent, so a caller whose label
changes is never ignored, but a turn spoken only by other voices is marked OTHER_VOICE. The
agent answers it if it is about cars or this enquiry, and otherwise stays silent (prompt:
OTHER VOICES, and the IGNORE reply in agent.llm_node).

Speechmatics sometimes relabels the caller mid-call, so two things are never marked: a
short reply ("No.", "Thank you.", "Saturday at six"), which is the caller answering the
agent, and a voice that has taken over the call, which becomes the caller.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("dasco")

UNKNOWN_SPEAKER = "UU"  # Speechmatics' label when it cannot tell
OTHER_VOICE = "[ANOTHER VOICE, not the caller]"

# A turn this short is the caller answering the agent, whatever its label.
SHORT_REPLY_WORDS = 3
# This many turns in a row from one other voice, and that voice is the caller now.
TAKEOVER_TURNS = 2


class SpeakerTracker:
    def __init__(self) -> None:
        self.caller: str | None = None
        self._turn: set[str | None] = set()  # None: a voice that could not be told apart
        self._turn_words = 0
        self._other: str | None = None  # the other voice that spoke the last turns alone
        self._other_turns = 0

    def attach(self, session: Any) -> None:
        session.on("user_input_transcribed", self.on_transcript)

    def on_transcript(self, ev: Any) -> None:
        if not ev.is_final or not ev.transcript.strip():
            return
        speaker = ev.speaker_id if ev.speaker_id not in (None, "", UNKNOWN_SPEAKER) else None
        if speaker and self.caller is None:
            self.caller = speaker
            logger.info("caller_voice_identified", extra={"speaker": speaker})
        self._turn.add(speaker)
        self._turn_words += len(ev.transcript.split())

    def turn_is_other_voice(self) -> bool:
        """Whether the turn just finished came only from voices other than the caller.

        Starts the next turn afresh. A voice that could not be told apart counts as the
        caller, so the caller is never marked by mistake on an unclear label.
        """
        speakers, self._turn = self._turn, set()
        words, self._turn_words = self._turn_words, 0
        if not speakers or self.caller is None:
            return False
        if self.caller in speakers or None in speakers:
            self._other, self._other_turns = None, 0
            return False
        voice = next(iter(speakers)) if len(speakers) == 1 else None
        if voice is not None and voice == self._other:
            self._other_turns += 1
        else:
            self._other, self._other_turns = voice, 1
        if voice is not None and self._other_turns >= TAKEOVER_TURNS:
            logger.info("caller_voice_changed", extra={"old": self.caller, "speaker": voice})
            self.caller, self._other, self._other_turns = voice, None, 0
            return False
        return words > SHORT_REPLY_WORDS
