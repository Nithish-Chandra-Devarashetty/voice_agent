"""What the agent can do during a call: look up cars, book test drives, save leads."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import aiohttp
from livekit.agents import RunContext, function_tool

import cars
from config import settings

logger = logging.getLogger("dasco")

IST = timezone(timedelta(hours=5, minutes=30))


@dataclass
class CallData:
    """What this call has captured so far, reachable from every tool as ``ctx.userdata``."""

    caller_phone: str | None = None  # from caller ID
    name: str | None = None
    car: str | None = None  # the model they came for
    has_old_car: bool | None = None
    old_car: str | None = None
    test_drive: str | None = None  # "Baleno on Wednesday 14 October at 05:30 PM"
    lead_saved: bool = False
    complaint_saved: bool = False

    def known_so_far(self) -> str:
        """One line restating the facts, so the agent never asks for them twice.

        It also carries the date and time, which keeps the system prompt identical on every
        call (and so cacheable) and keeps the clock current during a long call.
        """
        if self.has_old_car is None:
            exchange = "not asked"
        elif self.has_old_car:
            exchange = f"yes ({self.old_car or 'details not given'})"
        else:
            exchange = "no"
        return (
            f"KNOWN SO FAR: now={now():%A %d %B %Y, %I:%M %p} India time; "
            f"name={self.name or 'unknown'}; "
            f"phone={'known from caller ID' if self.caller_phone else 'unknown'}; "
            f"interested_car={self.car or 'unknown'}; old_car_exchange={exchange}; "
            f"test_drive={self.test_drive or 'not booked'}; "
            f"callback_saved={'yes' if self.lead_saved else 'no'}; "
            f"complaint_recorded={'yes' if self.complaint_saved else 'no'}"
        )


def now() -> datetime:
    return datetime.now(IST)


def normalize_phone(raw: str) -> str | None:
    """A 10-digit Indian number from "+91 98765-43210" and the like, or None."""
    digits = "".join(ch for ch in raw if ch.isdigit())
    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    elif len(digits) == 11 and digits.startswith("0"):
        digits = digits[1:]
    return digits if len(digits) == 10 else None


def mask_phone(phone: str | None) -> str:
    return f"{phone[:2]}******{phone[-2:]}" if phone and len(phone) >= 4 else "<none>"


def team_for(car: str | None, *, complaint: bool = False) -> str:
    """Who follows up: the NEXA or Arena sales team, or customer support."""
    if complaint:
        return "support"
    found = cars.find_car(car or "")
    return "arena" if found and found["brand"].lower() == "arena" else "nexa"


async def save_record(kind: str, record: dict[str, Any]) -> str:
    """Append to data/<kind>s.jsonl and POST to WEBHOOK_URL. Returns a reference."""
    reference = uuid.uuid4().hex[:6].upper()
    entry = {"type": kind, "reference": reference, "created_at": now().isoformat(), **record}
    folder = Path(settings.data_dir)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / f"{kind}s.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    if settings.webhook_url:
        try:
            timeout = aiohttp.ClientTimeout(total=5)
            async with (
                aiohttp.ClientSession(timeout=timeout) as http,
                http.post(settings.webhook_url, json=entry) as response,
            ):
                response.raise_for_status()
        except Exception as exc:  # noqa: BLE001 - any webhook failure is non-fatal
            # Saved locally already; the caller is not told about a webhook hiccup.
            logger.warning("webhook_failed", extra={"kind": kind, "error": str(exc)[:200]})
    logger.info("record_saved", extra={"kind": kind, "reference": reference})
    return reference


def _phone(ctx: RunContext[CallData], phone: str | None) -> str | None:
    # Some models send null, or words like "known", for a number they are not changing.
    if phone and any(ch.isdigit() for ch in phone):
        return normalize_phone(phone)
    return ctx.userdata.caller_phone


def _name(ctx: RunContext[CallData], name: str | None) -> str:
    return (name or "").strip() or ctx.userdata.name or ""


@function_tool
async def save_customer_name(ctx: RunContext[CallData], name: str) -> dict[str, Any]:
    """Save the caller's name as soon as they give it.

    Args:
        name: As heard, even if unusual.
    """
    ctx.userdata.name = name.strip() or ctx.userdata.name
    return {"saved": bool(ctx.userdata.name), "name": ctx.userdata.name}


@function_tool
async def save_exchange_details(
    ctx: RunContext[CallData], has_old_car: bool, old_car: str | None = None
) -> dict[str, Any]:
    """Save whether the caller has an old car to exchange.

    Args:
        has_old_car: Whether they want to exchange one.
        old_car: Its make, model and year, if said.
    """
    ctx.userdata.has_old_car = has_old_car
    ctx.userdata.old_car = ((old_car or "").strip() or None) if has_old_car else None
    return {"saved": True}


@function_tool
async def list_cars(ctx: RunContext[CallData], brand: str | None = "NEXA") -> dict[str, Any]:
    """Cars on sale with body type, seats, fuel, price and what they suit.

    Args:
        brand: "NEXA" or "Arena".
    """
    wanted = (brand or "").strip().lower() or "nexa"
    return {
        "cars": [
            {**cars.summary(car), "best_for": car.get("use_cases", [])}
            for car in cars.all_cars()
            if car.get("suggestable", True) and car["brand"].lower() == wanted
        ]
    }


@function_tool
async def get_car_details(ctx: RunContext[CallData], car: str) -> dict[str, Any]:
    """One Maruti Suzuki car's price, variants, features, mileage and offer types.

    Args:
        car: Model name, e.g. "Baleno".
    """
    found = cars.find_car(car)
    if found is None:
        return {
            "found": False,
            "note": "Not a Maruti Suzuki model we know. No facts available for it.",
            "known_models": [c["model"] for c in cars.all_cars()],
        }
    # The first car they ask about is the one they came for; later ones are comparisons.
    if ctx.userdata.car is None:
        ctx.userdata.car = found["model"]
    return {"found": True, **cars.details(found)}


@function_tool
async def book_test_drive(
    ctx: RunContext[CallData],
    car: str,
    date: str,
    time: str,
    name: str | None = None,
    phone: str | None = None,
) -> dict[str, Any]:
    """Book a NEXA test drive at the showroom.

    Args:
        car: NEXA model.
        date: YYYY-MM-DD.
        time: 24-hour HH:MM, e.g. "17:30".
        name: Only if not saved yet.
        phone: Only for a number other than caller ID.
    """
    found = cars.find_car(car)
    if found is None:
        return {"success": False, "problem": f"Unknown car {car!r}. Ask which model they mean."}
    if found["brand"].lower() != "nexa":
        return {
            "success": False,
            "problem": "Arena test drives are not booked on this line. Save a callback with "
            "save_lead so the Arena team contacts them.",
        }
    try:
        when = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M").replace(tzinfo=IST)
    except ValueError:
        return {"success": False, "problem": "Date must be YYYY-MM-DD and time HH:MM."}
    if when < now():
        return {"success": False, "problem": "That time has already passed. Ask for another."}
    minutes = when.hour * 60 + when.minute
    if not settings.showroom_open_hour * 60 <= minutes <= settings.showroom_close_hour * 60:
        return {"success": False, "problem": "The showroom is closed then. Offer opening hours."}
    number = _phone(ctx, phone)
    if number is None:
        return {"success": False, "problem": "Need a valid 10-digit phone number."}
    customer = _name(ctx, name)
    if not customer:
        return {"success": False, "problem": "Ask for their name first."}

    reference = await save_record(
        "test_drive",
        {
            "name": customer,
            "phone": number,
            "car": found["model"],
            "at": when.isoformat(),
            "old_car_exchange": ctx.userdata.old_car if ctx.userdata.has_old_car else None,
            "team": "nexa",
        },
    )
    spoken = when.strftime("%A %d %B at %I:%M %p")
    ctx.userdata.name, ctx.userdata.car = customer, found["model"]
    ctx.userdata.test_drive = f"{found['model']} on {spoken}"
    logger.info("test_drive_booked", extra={"car": found["model"], "phone": mask_phone(number)})
    return {"success": True, "car": found["model"], "when": spoken, "reference": reference}


@function_tool
async def save_lead(
    ctx: RunContext[CallData],
    interest: str,
    notes: str | None = None,
    name: str | None = None,
    phone: str | None = None,
) -> dict[str, Any]:
    """Save a callback request for the sales team (also Arena cars, on-road prices, offers).

    Args:
        interest: The car or topic.
        notes: Budget, timing, questions for the team.
        name: Only if not saved yet.
        phone: Only for a number other than caller ID.
    """
    number = _phone(ctx, phone)
    if number is None:
        return {"success": False, "problem": "Need a valid 10-digit phone number."}
    reference = await save_record(
        "lead",
        {
            "name": _name(ctx, name),
            "phone": number,
            "interest": interest.strip(),
            "notes": notes or "",
            "old_car_exchange": ctx.userdata.old_car if ctx.userdata.has_old_car else None,
            "team": team_for(interest),
        },
    )
    ctx.userdata.lead_saved = True
    return {"success": True, "team": team_for(interest), "reference": reference}


@function_tool
async def record_complaint(
    ctx: RunContext[CallData], details: str, name: str | None = None, phone: str | None = None
) -> dict[str, Any]:
    """Record a showroom complaint so customer support calls back.

    Args:
        details: What went wrong, in their words.
        name: Only if not saved yet.
        phone: Only for a number other than caller ID.
    """
    number = _phone(ctx, phone)
    if number is None:
        return {"success": False, "problem": "Need a valid 10-digit phone number."}
    reference = await save_record(
        "complaint",
        {"name": _name(ctx, name), "phone": number, "details": details, "team": "support"},
    )
    ctx.userdata.complaint_saved = True
    return {"success": True, "reference": reference}


ALL_TOOLS = [
    save_customer_name,
    save_exchange_details,
    list_cars,
    get_car_details,
    book_test_drive,
    save_lead,
    record_complaint,
]
