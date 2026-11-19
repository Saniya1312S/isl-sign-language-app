"""
Speech Service providing audio pronunciation data and browser speech metadata.
"""
from __future__ import annotations

import base64
import sys
from pathlib import Path
from typing import Dict, Any, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.speech import generate_audio_bytes, BROWSER_LOCALES


def get_speech_payload(text: str, language: str = "en") -> Dict[str, Any]:
    """
    Generate speech metadata:
    - Browser locale for SpeechSynthesisUtterance
    - Audio MP3 data URL (if gTTS available)
    """
    clean_text = (text or "").strip()
    if not clean_text:
        return {"status": "error", "message": "No text provided"}

    lang_lower = language.lower()
    if "marathi" in lang_lower or lang_lower == "mr":
        lang_code = "mr"
        locale = "mr-IN"
    elif "hindi" in lang_lower or lang_lower == "hi":
        lang_code = "hi"
        locale = "hi-IN"
    else:
        lang_code = "en"
        locale = "en-IN"

    # Try generating server audio as base64 data URL
    mp3_bytes, err = generate_audio_bytes(clean_text, lang=lang_code)
    audio_data_url = ""
    if mp3_bytes:
        b64_audio = base64.b64encode(mp3_bytes).decode("utf-8")
        audio_data_url = f"data:audio/mp3;base64,{b64_audio}"

    return {
        "status": "success",
        "text": clean_text,
        "language_code": lang_code,
        "browser_locale": locale,
        "audio_data_url": audio_data_url,
        "has_server_audio": bool(audio_data_url)
    }
