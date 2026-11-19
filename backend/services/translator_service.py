"""
Translation Service for converting English text to Hindi and Marathi.
Wraps src.translator with caching and structured output.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.translator import translate_text, LANG_CODES


def perform_translation(text: str, target_language: str = "hi") -> Dict[str, Any]:
    """
    Translate English text into target language (Hindi or Marathi).
    Returns structured translation response.
    """
    clean_text = (text or "").strip()
    if not clean_text:
        return {
            "original_text": "",
            "translated_text": "",
            "target_language": target_language,
            "source": "empty",
            "message": "Empty text provided"
        }

    lang_key = target_language.lower()
    if "marathi" in lang_key or lang_key == "mr":
        target = "mr"
        lang_name = "Marathi"
    elif "hindi" in lang_key or lang_key == "hi":
        target = "hi"
        lang_name = "Hindi"
    else:
        target = "en"
        lang_name = "English"

    res = translate_text(clean_text, target_lang=target, allow_online=True)
    return {
        "original_text": clean_text,
        "translated_text": res.text,
        "target_language": lang_name,
        "language_code": target,
        "source": res.source,
        "error": res.error
    }
