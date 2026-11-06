"""
SignAssistant - a short-answer Gemini assistant for people who communicate by fingerspelling.

Key design points (all fixes for the original app):
* Uses the current `google-genai` SDK (the old `google-generativeai` package was retired).
* The API key is NEVER in the source code: environment variable, Streamlit secrets or the sidebar.
* Nothing is called at import time or on every Streamlit rerun (the old app "probed" the API
  with a ping on each rerun). A request is made only when the user presses Send.
* The model name is configurable because Gemini model names change; `list_models()` can discover
  what the key can use.
* The client is injectable, so the logic is unit-tested without network access.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

DEFAULT_MODEL = "gemini-2.5-flash"

SYSTEM_PROMPT = """You are SignAssistant, a helper inside an Indian Sign Language (ISL) fingerspelling app.
The person talking to you is deaf or hard of hearing and types by signing letters, so their messages may be
short, lower-case and contain small spelling mistakes - interpret them kindly.
Rules:
- Answer in at most 3 short sentences, in plain simple words. No markdown, no bullet lists, no emojis.
- Reply in the language requested by the user message (English, Hindi or Marathi).
- Never invent facts. If you are not sure about something, say so.
- You are NOT a medical, legal or emergency service. For health or safety emergencies tell the person to ask
  a nearby person for help and to call 112 (India's emergency number).
- If asked how to make an ISL sign, only describe what you are confident about and suggest checking an official
  ISL dictionary (for example the ISLRTC dictionary) because you can make mistakes about handshapes."""


@dataclass
class AssistantReply:
    text: Optional[str]
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.text is not None


def resolve_api_key(session_key: str = "", secrets: Optional[dict] = None) -> str:
    """Key priority: sidebar field > Streamlit secrets > environment variable."""
    if session_key and session_key.strip():
        return session_key.strip()
    if secrets:
        for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            try:
                if secrets.get(name):
                    return str(secrets[name]).strip()
            except Exception:
                pass
    return (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()


def friendly_error(exc: Exception) -> str:
    msg = f"{type(exc).__name__}: {exc}"
    low = msg.lower()
    if any(k in low for k in ("api key", "api_key", "permission", "unauthenticated", "401", "403")):
        return "The API key was rejected. Check that it is correct and not revoked."
    if any(k in low for k in ("429", "quota", "resource_exhausted", "rate")):
        return "Rate limit or quota reached. Wait a minute and try again."
    if any(k in low for k in ("404", "not found", "not supported", "is not found")):
        return "That model name is not available for your key. Pick another model in the sidebar."
    if any(k in low for k in ("connect", "timeout", "network", "resolve", "unreachable")):
        return "No internet connection to the Gemini API."
    return msg[:200]


class SignAssistant:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, client=None,
                 max_output_tokens: int = 300, temperature: float = 0.4):
        self.model = model or DEFAULT_MODEL
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self._client = client
        self._api_key = api_key

    def _get_client(self):
        if self._client is None:
            from google import genai                       # imported lazily
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def ask(self, question: str, history: Optional[List[Tuple[str, str]]] = None,
            language: str = "English") -> AssistantReply:
        """`history` = [(role, text), ...] with role 'user' or 'assistant'."""
        question = (question or "").strip()
        if not question:
            return AssistantReply(None, "Type or sign a question first.")
        if not self._api_key and self._client is None:
            return AssistantReply(None, "Add a Gemini API key in the sidebar to use SignAssistant.")
        try:
            from google.genai import types
            contents = []
            for role, text in (history or [])[-6:]:
                contents.append(types.Content(role="user" if role == "user" else "model",
                                              parts=[types.Part(text=text)]))
            contents.append(types.Content(role="user", parts=[types.Part(
                text=f"[Reply in {language}] {question}")]))
            resp = self._get_client().models.generate_content(
                model=self.model, contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT, temperature=self.temperature,
                    max_output_tokens=self.max_output_tokens))
            text = (getattr(resp, "text", None) or "").strip()
            return AssistantReply(text) if text else AssistantReply(None, "The model returned an empty answer.")
        except Exception as exc:
            return AssistantReply(None, friendly_error(exc))

    def list_models(self) -> List[str]:
        """Names of Gemini text models this key can use (call only on user request)."""
        out = []
        for m in self._get_client().models.list():
            name = getattr(m, "name", "")
            actions = getattr(m, "supported_actions", None) or []
            if "gemini" in name and "generateContent" in actions and not any(
                    x in name for x in ("image", "tts", "live", "embedding", "audio", "robotics")):
                out.append(name.replace("models/", ""))
        return sorted(set(out))
