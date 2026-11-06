"""
English -> Hindi / Marathi.

Order of attempts:  1) built-in offline glossary   2) online Google Translate (deep-translator)
If both fail we return an explicit status so the UI can say so (the old app returned the literal
string "[Translation Error]" and then tried to speak it).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from .vocab import GLOSSARY, LETTER_NAMES

LANGS = {"English": "en", "Hindi": "hi", "Marathi": "mr"}


@dataclass
class Translation:
    text: Optional[str]
    source: str            # "glossary" | "online" | "unavailable" | "same"
    error: str = ""


def is_devanagari(text: str) -> bool:
    return any("\u0900" <= c <= "\u097F" for c in text)


def _glossary_lookup(text: str, lang: str) -> Optional[str]:
    key = text.strip().lower()
    idx = 0 if lang == "hi" else 1
    if key in GLOSSARY:
        return GLOSSARY[key][idx]
    if len(key) == 1 and key.upper() in LETTER_NAMES:
        return LETTER_NAMES[key.upper()]
    words = key.split()
    if len(words) > 1 and all(w in GLOSSARY for w in words):
        return " ".join(GLOSSARY[w][idx] for w in words)
    return None


@lru_cache(maxsize=512)
def _online(text: str, lang: str) -> tuple:
    try:
        from deep_translator import GoogleTranslator
        out = GoogleTranslator(source="en", target=lang).translate(text)
        return (out, "") if out else (None, "empty response")
    except Exception as exc:                       # network down, rate limit, ...
        return None, f"{type(exc).__name__}: {exc}"[:160]


def translate(text: str, lang: str, allow_online: bool = True) -> Translation:
    """`lang` is 'hi' or 'mr' ('en' returns the text unchanged)."""
    text = (text or "").strip()
    if not text or lang == "en" or is_devanagari(text):
        return Translation(text or None, "same")
    hit = _glossary_lookup(text, lang)
    if hit:
        return Translation(hit, "glossary")
    if allow_online:
        out, err = _online(text, lang)
        if out:
            return Translation(out, "online")
        return Translation(None, "unavailable", err)
    return Translation(None, "unavailable", "word not in the offline glossary")
