"""The Maruti Suzuki models the agent knows about (knowledge/*.json)."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
_FILLER_WORDS = {"maruti", "suzuki", "nexa", "arena", "the", "new"}


def _normalize(text: str) -> str:
    words = re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()
    return " ".join(w for w in words if w not in _FILLER_WORDS)


@lru_cache(maxsize=1)
def all_cars() -> list[dict[str, Any]]:
    cars: list[dict[str, Any]] = []
    for path in sorted(KNOWLEDGE_DIR.glob("*.json")):
        cars.extend(json.loads(path.read_text(encoding="utf-8"))["models"])
    return cars


def find_car(name: str) -> dict[str, Any] | None:
    """The car a caller means: by model name or a known mishearing ("balino")."""
    wanted = _normalize(name)
    if not wanted:
        return None
    for car in all_cars():
        names = [car["model"], *car.get("aliases", [])]
        if wanted in {_normalize(n) for n in names}:
            return car
    # "baleno alpha" or "grand vitara hybrid": a model name inside the phrase.
    for car in sorted(all_cars(), key=lambda c: -len(c["model"])):
        if f" {_normalize(car['model'])} " in f" {wanted} ":
            return car
    return None


def summary(car: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": car["model"],
        "brand": car["brand"],
        "body_type": car["body_type"],
        "seats": car["seating_capacity"],
        "fuel": car["fuel_options"],
        "price": car["pricing"]["spoken"],
    }


def details(car: dict[str, Any]) -> dict[str, Any]:
    mileage = {
        k: v for k, v in car.get("mileage", {}).items() if k not in {"basis", "as_of", "confidence"}
    }
    return {
        **summary(car),
        "price_note": "Approximate ex-showroom price; on-road price is higher.",
        "best_for": car.get("use_cases", []),
        "features": car.get("features", []),
        "variants": car.get("variants", []),
        "mileage_claimed": mileage,
        # Offer types only: amounts change monthly and are confirmed by sales.
        "offer_types": [offer["type"] for offer in car.get("offers", [])],
    }


def keyterms() -> list[str]:
    """Words Deepgram should listen for, so "Fronx" is not heard as "fronts"."""
    return ["Maruti Suzuki", "NEXA", "Arena", "test drive", *(c["model"] for c in all_cars())]
