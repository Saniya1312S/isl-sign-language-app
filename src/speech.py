"""
Lightweight, Reliable Text-to-Speech (TTS) for ISL Pronunciation.

Supports:
1. Browser Web Speech API:
   Client-side JavaScript SpeechSynthesis in HTML5 iframe.
   Runs offline, produces zero server load, works in Chrome, Edge, Safari, Firefox.
2. Server-side gTTS (Optional fallback):
   Generates MP3 bytes for Streamlit st.audio player if desired.
"""
from __future__ import annotations

import html
import io
import json
from typing import Optional, Tuple

BROWSER_LOCALES = {
    "en": "en-IN",
    "English": "en-IN",
    "hi": "hi-IN",
    "Hindi": "hi-IN",
    "mr": "mr-IN",
    "Marathi": "mr-IN"
}


def generate_browser_speech_html(text: str, lang: str = "en", rate: float = 0.95, nonce: int = 0) -> str:
    """
    Generate client-side HTML/JS component using the Web Speech API.
    Streamlit renders this in an iframe to trigger native speech pronunciation.
    """
    clean_text = (text or "").strip()
    if not clean_text:
        return ""

    locale = BROWSER_LOCALES.get(lang, "en-IN")
    payload = json.dumps({"text": clean_text, "lang": locale, "rate": rate}).replace("</", "<\\/")

    return f"""
<div id="speech_status" style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-size: 0.85rem; color: #14b8a6; padding: 6px 12px; background: #0f172a; border-radius: 6px; display: inline-flex; align-items: center; gap: 8px;">
    <span>🔊 Pronouncing...</span>
    <span style="display:none">{nonce}</span>
</div>
<script>
(function() {{
    const p = {payload};
    const statusEl = document.getElementById('speech_status');
    if (!('speechSynthesis' in window)) {{
        statusEl.innerHTML = '<span style="color:#ef4444;">⚠️ Browser speech synthesis not supported</span>';
        return;
    }}
    const speak = () => {{
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(p.text);
        utterance.lang = p.lang;
        utterance.rate = p.rate;
        const voices = window.speechSynthesis.getVoices();
        const matchedVoice = voices.find(v => v.lang === p.lang) || voices.find(v => v.lang.startsWith(p.lang.slice(0, 2)));
        if (matchedVoice) {{
            utterance.voice = matchedVoice;
        }}
        utterance.onend = () => {{
            statusEl.innerHTML = '<span style="color:#10b981;">✔ Pronunciation complete</span>';
        }};
        utterance.onerror = (e) => {{
            statusEl.innerHTML = '<span style="color:#ef4444;">⚠️ Speech error: ' + (e.error || 'Playback blocked') + '</span>';
        }};
        window.speechSynthesis.speak(utterance);
    }};
    if (window.speechSynthesis.getVoices().length) {{
        speak();
    }} else {{
        window.speechSynthesis.onvoiceschanged = speak;
    }}
}})();
</script>
"""


def generate_audio_bytes(text: str, lang: str = "en") -> Tuple[Optional[bytes], str]:
    """
    Generate MP3 audio bytes using gTTS if installed and connected.
    Returns (mp3_bytes, error_message).
    """
    clean_text = (text or "").strip()
    if not clean_text:
        return None, "Empty text to pronounce"

    lang_code = "hi" if lang.lower().startswith("hi") else ("mr" if lang.lower().startswith("mr") else "en")

    try:
        from gtts import gTTS
        buf = io.BytesIO()
        tld = "co.in" if lang_code == "en" else "com"
        tts = gTTS(text=clean_text, lang=lang_code, tld=tld, timeout=3)
        tts.write_to_fp(buf)
        return buf.getvalue(), ""
    except Exception as ex:
        return None, str(ex)
