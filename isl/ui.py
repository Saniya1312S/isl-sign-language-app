"""Shared look & feel (CSS) and small reusable UI helpers."""
import re

import streamlit as st
import streamlit.components.v1 as components

from .voice import browser_speech_html, tts_mp3

CSS = """
<style>
.block-container {padding-top: 1.6rem; max-width: 1250px;}
h1, h2, h3 {letter-spacing: -0.01em;}
.hero {background: linear-gradient(120deg,#0f766e 0%,#0e7490 45%,#1e3a8a 100%); border-radius: 18px;
       padding: 1.6rem 1.8rem; margin-bottom: 1.1rem; color: #f0fdfa;}
.hero h1 {margin: 0 0 .3rem 0; font-size: 2rem; color: #fff;}
.hero p {margin: 0; opacity: .92; font-size: 1.02rem;}
.kpi {background:#131C2E; border:1px solid #22304a; border-radius:14px; padding:.9rem 1rem;}
.kpi .v {font-size:1.7rem; font-weight:700; color:#5eead4; line-height:1.1;}
.kpi .l {font-size:.78rem; text-transform:uppercase; letter-spacing:.06em; color:#94a3b8;}
.kpi .s {font-size:.78rem; color:#94a3b8; margin-top:.15rem;}
.typed {font-size: 2.4rem; font-weight: 700; letter-spacing: .08em; min-height: 3.4rem; padding: .55rem .9rem;
        background:#0f172a; border:2px solid #14B8A6; border-radius:14px; color:#f8fafc; word-break: break-word;}
.cand {font-size:.95rem; color:#94a3b8;}
.note {border-left: 4px solid #f59e0b; background:#2a2110; padding:.6rem .9rem; border-radius:8px; color:#fde68a; font-size:.92rem;}
.good {border-left: 4px solid #14B8A6; background:#0b2a2a; padding:.6rem .9rem; border-radius:8px; color:#99f6e4; font-size:.92rem;}
.chip {display:inline-block; padding:.15rem .6rem; border-radius:999px; background:#1e293b; color:#cbd5e1; font-size:.8rem; margin-right:.3rem;}
</style>
"""


def md(text: str) -> str:
    """Tiny markdown subset for our HTML blocks: **bold** and `code`."""
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", text)


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{md(subtitle)}</p></div>', unsafe_allow_html=True)


def kpi(col, value: str, label: str, sub: str = "") -> None:
    col.markdown(f'<div class="kpi"><div class="v">{value}</div><div class="l">{label}</div>'
                 f'<div class="s">{sub}</div></div>', unsafe_allow_html=True)


def note(text: str, kind: str = "note") -> None:
    st.markdown(f'<div class="{kind}">{md(text)}</div>', unsafe_allow_html=True)


def speak(slot, text: str, lang: str, engine: str) -> None:
    """Speak `text`. Rendered into `slot`, a container created in the main script, so the audio
    element is not wiped by the 0.5 s auto-refresh of the text fragment."""
    if not text.strip():
        slot.warning("Nothing to speak yet.")
        return
    with slot:
        if engine.startswith("Browser"):
            st.session_state["speech_nonce"] = st.session_state.get("speech_nonce", 0) + 1
            components.html(browser_speech_html(text, lang, nonce=st.session_state["speech_nonce"]), height=34)
        else:
            mp3, err = tts_mp3(text, lang)
            if mp3:
                st.audio(mp3, format="audio/mp3", autoplay=True)
            else:
                st.warning(f"Server voice unavailable ({err}). Try the browser voice instead.")
