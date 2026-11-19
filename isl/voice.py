"""
Speech output. Two independent engines, so voice works locally AND on a cloud server:

1. Browser speech (Web Speech API)  - runs in the user's browser, no server, works offline in
   Chrome/Edge if the language voice is installed. Triggered by rendering a tiny HTML component.
2. gTTS (Google Text-to-Speech)     - the server makes an MP3 (needs internet) and Streamlit plays it.
   Reliable for Hindi and Marathi on any device.

The old app used `pyttsx3`, which speaks on the SERVER machine only (Windows SAPI) - it cannot
work when the app is deployed. Neither engine here has that limitation.
"""
from __future__ import annotations

import html
import io
import json
from functools import lru_cache
from typing import Optional, Tuple

from .vocab import LETTER_NAMES

BROWSER_LOCALE = {"en": "en-IN", "hi": "hi-IN", "mr": "mr-IN"}


def spell_out(text: str, lang: str = "en") -> str:
    """'CAT' -> 'C. A. T.' so the voice reads letter names (Devanagari names for hi/mr)."""
    letters = [c for c in text.upper() if c.isalpha() or c.isdigit()]
    if lang in ("hi", "mr"):
        return ". ".join(LETTER_NAMES.get(c, c) for c in letters) + "."
    return ". ".join(letters) + "."


@lru_cache(maxsize=128)
def tts_mp3(text: str, lang: str = "en") -> Tuple[Optional[bytes], str]:
    """Return (mp3_bytes, error). Cached so repeating a phrase is instant."""
    text = (text or "").strip()
    if not text:
        return None, "nothing to speak"
    try:
        from gtts import gTTS
        buf = io.BytesIO()
        kwargs = {"tld": "co.in"} if lang == "en" else {}        # Indian English accent
        gTTS(text=text, lang=lang, **kwargs).write_to_fp(buf)
        return buf.getvalue(), ""
    except Exception as exc:                                     # offline, blocked, unsupported
        return None, f"{type(exc).__name__}: {exc}"[:160]


def browser_speech_html(text: str, lang: str = "en", rate: float = 0.95, nonce: int = 0) -> str:
    """HTML snippet for st.components.v1.html(...) that speaks `text` in the browser.

    `nonce` changes on every request so Streamlit re-renders the iframe and the speech repeats.
    """
    # "</" is escaped so typed text can never close the <script> tag early
    payload = json.dumps({"text": text, "lang": BROWSER_LOCALE.get(lang, "en-IN"), "rate": rate}).replace("</", "<\\/")
    return f"""
<div id="s" style="font:13px sans-serif;color:#6b7280">🔊 speaking… <span style="display:none">{nonce}</span></div>
<script>
(function() {{
  const p = {payload};
  const el = document.getElementById('s');
  if (!('speechSynthesis' in window)) {{ el.textContent = '⚠ This browser has no speech synthesis'; return; }}
  const speak = () => {{
    const u = new SpeechSynthesisUtterance(p.text);
    u.lang = p.lang; u.rate = p.rate;
    const voices = window.speechSynthesis.getVoices();
    const v = voices.find(x => x.lang === p.lang) || voices.find(x => x.lang.startsWith(p.lang.slice(0,2)));
    if (v) u.voice = v; else el.textContent = '⚠ No ' + p.lang + ' voice installed - try the server voice';
    u.onend = () => {{ if (v) el.textContent = '✔ done'; }};
    window.speechSynthesis.cancel(); window.speechSynthesis.speak(u);
  }};
  if (window.speechSynthesis.getVoices().length) speak();
  else window.speechSynthesis.onvoiceschanged = speak;
}})();
</script>
"""
