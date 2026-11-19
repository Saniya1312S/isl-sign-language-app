"""
Translation and Speech API Endpoints.
Provides:
- POST /api/translate: English -> Hindi / Marathi translation
- POST /api/speak: Pronunciation payload for browser speech and server audio
"""
from __future__ import annotations

from typing import Any, Dict, Optional
from fastapi import APIRouter
from pydantic import BaseModel

from backend.services.translator_service import perform_translation
from backend.services.speech_service import get_speech_payload

router = APIRouter(prefix="/api", tags=["Translation & Speech"])


class TranslateRequest(BaseModel):
    text: str
    target_language: str = "Hindi"


class SpeakRequest(BaseModel):
    text: str
    language: str = "English"


@router.post("/translate")
def translate_endpoint(req: TranslateRequest) -> Dict[str, Any]:
    """Translate English sentence to Hindi or Marathi."""
    return perform_translation(req.text, target_language=req.target_language)


@router.post("/speak")
def speak_endpoint(req: SpeakRequest) -> Dict[str, Any]:
    """Generate pronunciation speech payload."""
    return get_speech_payload(req.text, language=req.language)
