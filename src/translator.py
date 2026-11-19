"""
Lightweight Translation Engine for ISL Alphabet fingerspelling sentences.
Translates English text/words to Hindi and Marathi.

Features:
- 100% offline dictionary & glossary for common words, single letters, and phrases
- Letter-by-letter phonetic spelling in Devanagari (A -> ए, B -> बी, etc.)
- Optional online translation (deep_translator) if available and network is active
- Fully zero-dependency offline fallback (requires NO Gemini and NO external API)
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

LANG_CODES = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr"
}

# Devanagari phonetic names for ISL single-letter signs
LETTER_NAMES: Dict[str, str] = {
    "A": "ए", "B": "बी", "C": "सी", "D": "डी", "E": "ई", "F": "एफ़",
    "G": "जी", "H": "एच", "I": "आई", "J": "जे", "K": "के", "L": "एल",
    "M": "एम", "N": "एन", "O": "ओ", "P": "पी", "Q": "क्यू", "R": "आर",
    "S": "एस", "T": "टी", "U": "यू", "V": "वी", "W": "डब्ल्यू", "X": "एक्स",
    "Y": "वाई", "Z": "ज़ेड"
}

# Offline Core Glossary: English -> (Hindi, Marathi)
OFFLINE_GLOSSARY: Dict[str, Tuple[str, str]] = {
    # Greetings & Common Expressions
    "hello": ("नमस्ते", "नमस्कार"),
    "hi": ("नमस्ते", "नमस्कार"),
    "thank you": ("धन्यवाद", "धन्यवाद"),
    "thanks": ("धन्यवाद", "धन्यवाद"),
    "welcome": ("स्वागत है", "स्वागत आहे"),
    "please": ("कृपया", "कृपया"),
    "sorry": ("माफ़ कीजिए", "माफ करा"),
    "goodbye": ("अलविदा", "निरोप"),
    "bye": ("अलविदा", "निरोप"),
    "yes": ("हाँ", "हो"),
    "no": ("नहीं", "नाही"),
    "ok": ("ठीक है", "ठीक आहे"),
    "good": ("अच्छा", "चांगला"),
    "bad": ("बुरा", "वाईट"),
    
    # Needs & Actions
    "help": ("मदद", "मदत"),
    "water": ("पानी", "पाणी"),
    "food": ("खाना", "जेवण"),
    "eat": ("खाओ", "खा"),
    "drink": ("पियो", "प्या"),
    "sleep": ("सोना", "झोपणे"),
    "stop": ("रुको", "थांबा"),
    "come": ("आओ", "ये"),
    "go": ("जाओ", "जा"),
    "wait": ("रुको", "थांबा"),
    "need": ("चाहिए", "हवे"),
    "work": ("काम", "काम"),
    "read": ("पढ़ो", "वाचा"),
    "write": ("लिखो", "लिहा"),
    "see": ("देखो", "बघा"),
    "look": ("देखो", "बघा"),

    # Health & Emergency
    "doctor": ("डॉक्टर", "डॉक्टर"),
    "hospital": ("अस्पताल", "रुग्णालय"),
    "medicine": ("दवा", "औषध"),
    "pain": ("दर्द", "वेदना"),
    "sick": ("बीमार", "आजारी"),
    "fever": ("बुखार", "ताप"),
    "emergency": ("आपातकाल", "आणीबाणी"),
    "police": ("पुलिस", "पोलीस"),

    # People & Family
    "mother": ("माँ", "आई"),
    "father": ("पिता", "वडील"),
    "friend": ("दोस्त", "मित्र"),
    "brother": ("भाई", "भाऊ"),
    "sister": ("बहन", "बहीण"),
    "family": ("परिवार", "कुटुंब"),
    "teacher": ("शिक्षक", "शिक्षक"),
    "student": ("विद्यार्थी", "विद्यार्थी"),
    "deaf": ("बधिर", "कर्णबधिर"),
    
    # Places & Things
    "home": ("घर", "घर"),
    "house": ("घर", "घर"),
    "school": ("स्कूल", "शाळा"),
    "name": ("नाम", "नाव"),
    "book": ("किताब", "पुस्तक"),
    "pen": ("कलम", "पेन"),
    "time": ("समय", "वेळ"),
    "today": ("आज", "आज"),
    "tomorrow": ("कल", "उद्या"),
    "yesterday": ("कल", "काल"),

    # Pronouns & Questions
    "i": ("मैं", "मी"),
    "you": ("आप", "तुम्ही"),
    "we": ("हम", "आम्ही"),
    "what": ("क्या", "काय"),
    "where": ("कहाँ", "कुठे"),
    "who": ("कौन", "कोण"),
    "why": ("क्यों", "का"),
    "how": ("कैसे", "कसे")
}


@dataclass
class TranslationResult:
    text: str
    target_language: str
    source: str  # "same", "glossary", "phonetic", "online", "unavailable"
    error: str = ""


def is_devanagari(text: str) -> bool:
    """Check if string already contains Devanagari characters."""
    return any("\u0900" <= c <= "\u097F" for c in text)


def translate_text(text: str, target_lang: str = "hi", allow_online: bool = True) -> TranslationResult:
    """
    Translate English text to Hindi ('hi') or Marathi ('mr').
    Returns TranslationResult with translated text and method used.
    """
    text = (text or "").strip()
    if not text:
        return TranslationResult(text="", target_language=target_lang, source="same")

    # If target is English or already in Devanagari, return as-is
    if target_lang in ("en", "English") or is_devanagari(text):
        return TranslationResult(text=text, target_language=target_lang, source="same")

    lang_key = "hi" if target_lang.lower().startswith("hi") else ("mr" if target_lang.lower().startswith("mr") else "hi")
    idx = 0 if lang_key == "hi" else 1

    clean_lower = text.lower()

    # 1. Direct whole phrase / word match
    if clean_lower in OFFLINE_GLOSSARY:
        return TranslationResult(
            text=OFFLINE_GLOSSARY[clean_lower][idx],
            target_language=target_lang,
            source="glossary"
        )

    # 2. Single letter match
    if len(text) == 1 and text.upper() in LETTER_NAMES:
        return TranslationResult(
            text=LETTER_NAMES[text.upper()],
            target_language=target_lang,
            source="phonetic"
        )

    # 3. Word-by-word glossary translation
    words = clean_lower.split()
    translated_words = []
    all_glossary = True
    for w in words:
        clean_w = w.strip(".,!?:;")
        if clean_w in OFFLINE_GLOSSARY:
            translated_words.append(OFFLINE_GLOSSARY[clean_w][idx])
        elif len(clean_w) == 1 and clean_w.upper() in LETTER_NAMES:
            translated_words.append(LETTER_NAMES[clean_w.upper()])
        else:
            all_glossary = False
            translated_words.append(w)

    if all_glossary:
        return TranslationResult(
            text=" ".join(translated_words),
            target_language=target_lang,
            source="glossary"
        )

    # 4. Optional online fallback if enabled
    if allow_online:
        try:
            from deep_translator import GoogleTranslator
            res = GoogleTranslator(source="en", target=lang_key).translate(text)
            if res:
                return TranslationResult(text=res, target_language=target_lang, source="online")
        except Exception as ex:
            pass  # Fall through to best-effort word translation

    # 5. Best-effort offline translation with available glossary words
    return TranslationResult(
        text=" ".join(translated_words),
        target_language=target_lang,
        source="glossary_partial"
    )
