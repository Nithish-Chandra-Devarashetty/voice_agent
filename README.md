# Phone agent (v3)

A small LiveKit voice agent that answers dasco Motors' phone line in English.

- **Speech-to-text:** Speechmatics with speaker diarization: another voice near the phone is
  answered only if it asks about cars; anything else it says is ignored
- **Voice:** Deepgram Aura-2, using your Deepgram API key
- **LLM:** Claude Haiku 5.5, with prompt caching, low effort and thinking off for fast replies
- **Does:** answers questions about Maruti Suzuki cars (NEXA and Arena), books test drives,
  saves callback requests, uses the caller's phone number automatically, ends the call itself
  when the caller is done, and hangs up after 30 seconds of silence.

## Files

| File | What it is |
|---|---|
| `agent.py` | The call: Speechmatics, Deepgram voice, Claude LLM, turn detection, greeting, end call, silence hang-up |
| `prompts.py` | What the agent is told (English only, call rules) |
| `tools.py` | Car lookup, test-drive booking, callback leads, saving records |
| `history.py` | Keeps each LLM request small: recent turns verbatim, older ones summarised |
| `speakers.py` | Diarization: marks turns spoken by someone other than the caller |
| `claude_request.py` | Sets effort and thinking on every Claude request |
| `usage.py` | Logs every Claude request: tokens, cache use, estimated cost, latency, rate limits |
| `cars.py` + `knowledge/` | The car catalogue (indicative prices, ex-showroom Delhi) |
| `config.py` | Settings from `.env` |

## Run it locally

```
py -3.11 -m venv .venv
.venv\Scripts\pip install -r requirements-dev.txt
copy .env.example .env          # then fill in the keys
.venv\Scripts\python agent.py download-files
.venv\Scripts\python agent.py console    # talk to it with your mic, no LiveKit room needed
```

`python agent.py dev` connects it to your LiveKit project instead. It is a named agent
(`AGENT_NAME`), so it only joins rooms dispatched to that name: test it from the LiveKit
Agent Console or a phone number pointed at it.

Tests: `.venv\Scripts\python -m pytest`

## Deploy to LiveKit Cloud

1. `copy .env .env.cloud`, then delete the three `LIVEKIT_` lines from `.env.cloud`.
2. Create the agent (first time only):
   ```
   lk agent create --secrets-file .env.cloud --ignore-empty-secrets .
   ```
   This writes the agent's ID to `livekit.toml`.
3. Later updates: `lk agent deploy`
4. Check: `lk agent status` and `lk agent logs`

## Connect a phone number

In the LiveKit Cloud dashboard → Telephony:
1. Buy a number (or `lk number search` / `lk number purchase`).
2. Create a dispatch rule that puts each caller in their own room and dispatches the agent
   named **`dasco-v3`** (your `AGENT_NAME`), and attach the number to it.

## Token use

- The system prompt and tools are identical on every call, so Claude's prompt cache serves
  them (`cached_tokens` in the logs). The date, time and what the call has captured so far
  come each turn in a short `KNOWN SO FAR` line at the end instead.
- Once a call passes 6 customer turns, older turns are summarised by Claude
  in the background and only the last 3 are sent word for word.
- Each request logs an `llm_request` line (prompt, cached, cache-write and completion
  tokens, estimated cost in dollars, time to first token, remaining rate limit); each call
  ends with `llm_call_totals`.

## Bookings and leads

Each one is appended to `data/test_drives.jsonl` or `data/leads.jsonl`, and POSTed as JSON to
`WEBHOOK_URL` if set. On LiveKit Cloud the container's files are lost on restart, so set
`WEBHOOK_URL` there (your CRM, a Zapier hook, a Google Apps Script...).
