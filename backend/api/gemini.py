"""
Gemini AI Assistant API Endpoints.
Provides:
- GET  /api/gemini/status: Comprehensive diagnostic test of Gemini API, SDK, and key
- POST /api/gemini/chat: Conversational completion with context and multilingual support
- POST /api/gemini/validate: Validate provided API key
- POST /api/gemini/set-key: Set runtime session key in memory
- POST /api/gemini/clear-key: Clear runtime session key
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Query
from pydantic import BaseModel

from backend.services.gemini_service import (
    call_gemini,
    validate_api_key,
    get_gemini_status,
    set_session_api_key,
    clear_session_api_key
)

router = APIRouter(prefix="/api/gemini", tags=["Gemini AI Assistant"])


class ChatMessage(BaseModel):
    role: str  # "user" or "model"
    text: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = None
    language: Optional[str] = "English"
    api_key: Optional[str] = None


class ValidateKeyRequest(BaseModel):
    api_key: str


class SetKeyRequest(BaseModel):
    api_key: str


@router.get("/status")
def gemini_status_endpoint(api_key: Optional[str] = Query(None)) -> Dict[str, Any]:
    """
    Diagnostic status endpoint reporting:
    - Whether a key is configured
    - Whether the Gemini SDK is available
    - Whether the model can be reached
    - Whether the connection test succeeded
    Never returns the actual API key.
    """
    return get_gemini_status(api_key=api_key)


@router.post("/set-key")
def gemini_set_key_endpoint(req: SetKeyRequest) -> Dict[str, Any]:
    """Store runtime session key in backend memory (never saved to disk or frontend)."""
    set_session_api_key(req.api_key)
    # Validate immediately
    status = get_gemini_status(api_key=req.api_key)
    return {
        "status": "success" if status.get("connection_success") else "warning",
        "key_configured": status.get("key_configured", False),
        "connection_success": status.get("connection_success", False),
        "message": status.get("message", "Session key updated.")
    }


@router.post("/clear-key")
def gemini_clear_key_endpoint() -> Dict[str, Any]:
    """Clear in-memory session key."""
    clear_session_api_key()
    return {"status": "success", "message": "Session key cleared."}


@router.post("/chat")
def gemini_chat_endpoint(req: ChatRequest) -> Dict[str, Any]:
    """Execute Gemini chat interaction with conversational history and language option."""
    history_dicts = [{"role": h.role, "text": h.text} for h in (req.history or [])]
    return call_gemini(
        message=req.message,
        history=history_dicts,
        language=req.language or "English",
        api_key=req.api_key
    )


@router.post("/validate")
def gemini_validate_endpoint(req: ValidateKeyRequest) -> Dict[str, Any]:
    """Test validity of a Gemini API key without exposing the key."""
    return validate_api_key(req.api_key)
