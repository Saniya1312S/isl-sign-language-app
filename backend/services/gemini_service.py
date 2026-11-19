"""
Gemini Service for Conversational AI Assistant.
Provides both official Google GenAI SDK (google-genai) integration
and resilient REST fallback, comprehensive diagnostics, session-memory key management,
and multilingual prompt orchestration.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
import requests

from backend.config import get_gemini_api_key

logger = logging.getLogger("gemini_service")

# Supported models in priority order
GEMINI_MODELS = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro"
]

# Check SDK availability
try:
    from google import genai
    from google.genai import types as genai_types
    HAS_GOOGLE_GENAI = True
    SDK_VERSION = getattr(genai, "__version__", "3.6.1")
except Exception as ex:
    HAS_GOOGLE_GENAI = False
    SDK_VERSION = None
    genai = None
    genai_types = None

# In-memory session key (never written to disk or sent to frontend)
_SESSION_API_KEY: Optional[str] = None


def set_session_api_key(key: str) -> None:
    """Store an in-memory session API key for the current runtime."""
    global _SESSION_API_KEY
    _SESSION_API_KEY = key.strip() if key else None


def clear_session_api_key() -> None:
    """Clear the in-memory session API key."""
    global _SESSION_API_KEY
    _SESSION_API_KEY = None


def resolve_active_key(override_key: Optional[Any] = None) -> Tuple[str, str]:
    """
    Resolve active API key and report its origin.
    Returns (key, source):
    - source can be 'explicit', 'session', 'environment', or 'none'.
    """
    if isinstance(override_key, str) and override_key.strip():
        return override_key.strip(), "explicit"
    if _SESSION_API_KEY:
        return _SESSION_API_KEY, "session"
    env_key = get_gemini_api_key()
    if env_key:
        return env_key, "environment"
    return "", "none"


def mask_key(key: str) -> str:
    """Mask key for safe diagnostic reporting (e.g. AIzaSy...9xyz)."""
    if not key:
        return ""
    if len(key) <= 8:
        return "****"
    return f"{key[:6]}...{key[-4:]}"


def parse_gemini_error(raw_error: str, status_code: Optional[int] = None) -> str:
    """Translate raw error strings / HTTP codes into human-understandable guidance."""
    err_str = str(raw_error)
    err_lower = err_str.lower()

    if status_code == 400 or "api_key_invalid" in err_lower or "api key not valid" in err_lower:
        return "Invalid Gemini API key. Please check that you copied the key correctly from Google AI Studio."
    if status_code == 403 or "permission_denied" in err_lower or "unregistered" in err_lower:
        return "Gemini API permission denied. Ensure the Generative Language API is enabled for this Google Cloud project."
    if status_code == 429 or "resource_exhausted" in err_lower or "quota" in err_lower or "rate limit" in err_lower:
        return "Gemini API rate limit or quota exceeded. Please wait a minute before sending another request."
    if status_code == 404 or "model not found" in err_lower:
        return "The requested Gemini model is currently unavailable for this API version."
    if "timeout" in err_lower or "timed out" in err_lower:
        return "Connection timed out while reaching Google Gemini servers. Please check your internet connection."
    if "connection" in err_lower or "failed to establish" in err_lower or "getaddrinfo" in err_lower:
        return "Network error: Unable to reach Google Gemini API endpoints. Please verify internet connectivity."

    # Generic clean fallback
    clean = re.sub(r"[\n\r\t]+", " ", err_str).strip()
    if len(clean) > 200:
        clean = clean[:200] + "..."
    return f"Gemini communication error: {clean}"


def build_system_prompt(language: str) -> str:
    """Generate localized system instructions for the assistant."""
    lang = (language or "English").lower()
    if "hindi" in lang:
        lang_instruction = "Respond entirely in Hindi (हिंदी) using Devanagari script."
    elif "marathi" in lang:
        lang_instruction = "Respond entirely in Marathi (मराठी) using Devanagari script."
    else:
        lang_instruction = "Respond in clear, natural English."

    return (
        "You are an empathetic, articulate, and supportive conversational AI assistant for an Indian Sign Language (ISL) recognition website. "
        "The user communicates via ISL fingerspelling or gesture translation. "
        f"{lang_instruction} "
        "Keep your response natural, helpful, friendly, and concise (within 2-3 sentences)."
    )


def call_gemini(message: str,
                history: Optional[List[Dict[str, str]]] = None,
                language: str = "English",
                api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Execute Gemini conversation request with official SDK and REST fallback.
    Never exposes raw API keys.
    """
    active_key, key_source = resolve_active_key(api_key)
    if not active_key:
        return {
            "status": "no_key",
            "reply": "⚠️ Gemini API key is not configured. You can enter your API key in the connection box above or add GEMINI_API_KEY in your server's .env file.",
            "language": language,
            "key_configured": False
        }

    clean_message = (message or "").strip()
    if not clean_message:
        return {
            "status": "error",
            "reply": "Please enter or sign a message to send to the Gemini assistant.",
            "language": language,
            "key_configured": True
        }

    system_prompt = build_system_prompt(language)

    # 1. Try official google-genai SDK first if installed
    if HAS_GOOGLE_GENAI and genai is not None:
        for model_name in GEMINI_MODELS:
            try:
                client = genai.Client(api_key=active_key)
                # Build contents
                contents_parts: List[Any] = []
                # Provide history
                if history:
                    for turn in history[-6:]:
                        role = "user" if turn.get("role") == "user" else "model"
                        t = turn.get("text", "")
                        if t:
                            contents_parts.append(f"{role.capitalize()}: {t}")

                contents_parts.append(f"User: {clean_message}")
                full_prompt = f"[System Context: {system_prompt}]\n\n" + "\n".join(contents_parts)

                config = None
                if genai_types:
                    config = genai_types.GenerateContentConfig(
                        temperature=0.7,
                        max_output_tokens=600
                    )

                response = client.models.generate_content(
                    model=model_name,
                    contents=full_prompt,
                    config=config
                )

                if response and response.text:
                    return {
                        "status": "success",
                        "reply": response.text.strip(),
                        "language": language,
                        "model": model_name,
                        "client_type": "official_sdk",
                        "key_source": key_source
                    }
            except Exception as ex:
                logger.warning(f"google-genai SDK call failed with model {model_name}: {ex}")
                # If it's an invalid key or permission denied, stop retrying models
                err_msg = parse_gemini_error(str(ex))
                if "Invalid Gemini API key" in err_msg or "permission denied" in err_msg:
                    return {
                        "status": "api_error",
                        "reply": f"⚠️ {err_msg}",
                        "language": language,
                        "key_configured": True,
                        "error_details": str(ex)
                    }

    # 2. Resilient REST API fallback
    last_error = ""
    for model_name in GEMINI_MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={active_key}"
        
        contents: List[Dict[str, Any]] = []
        if history:
            for turn in history[-6:]:
                role = "user" if turn.get("role") == "user" else "model"
                txt = turn.get("text", "")
                if txt:
                    contents.append({"role": role, "parts": [{"text": txt}]})

        user_content = f"[System Context: {system_prompt}]\n\nUser: {clean_message}"
        contents.append({"role": "user", "parts": [{"text": user_content}]})

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 600
            }
        }

        try:
            resp = requests.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=18
            )
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        reply_text = parts[0].get("text", "").strip()
                        return {
                            "status": "success",
                            "reply": reply_text,
                            "language": language,
                            "model": model_name,
                            "client_type": "rest_fallback",
                            "key_source": key_source
                        }
            else:
                err_body = resp.text
                last_error = parse_gemini_error(err_body, status_code=resp.status_code)
                if resp.status_code in (400, 403):
                    break
        except requests.exceptions.Timeout:
            last_error = "Connection timed out while reaching Google Gemini servers."
            break
        except Exception as ex:
            last_error = parse_gemini_error(str(ex))

    return {
        "status": "api_error",
        "reply": f"⚠️ {last_error or 'Could not complete request to Gemini.'}",
        "language": language,
        "key_configured": True,
        "error_details": last_error
    }


def validate_api_key(api_key: str) -> Dict[str, Any]:
    """
    Test validity of a Gemini API key using a minimal 1-word generation probe.
    Never logs or exposes the raw key.
    """
    key = (api_key or "").strip()
    if not key:
        return {
            "valid": False,
            "status": "missing_key",
            "message": "API key cannot be empty."
        }

    # Test via official SDK first if available
    if HAS_GOOGLE_GENAI and genai is not None:
        try:
            client = genai.Client(api_key=key)
            resp = client.models.generate_content(
                model="gemini-2.0-flash",
                contents="Hi"
            )
            if resp and resp.text:
                return {
                    "valid": True,
                    "status": "connected",
                    "model": "gemini-2.0-flash",
                    "message": "Gemini API key verified successfully via official Google GenAI SDK."
                }
        except Exception as ex:
            err_msg = parse_gemini_error(str(ex))
            if "Invalid Gemini API key" in err_msg or "permission denied" in err_msg:
                return {
                    "valid": False,
                    "status": "auth_failed",
                    "message": err_msg
                }

    # Test via REST
    for model_name in ["gemini-2.0-flash", "gemini-1.5-flash"]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={key}"
        payload = {"contents": [{"role": "user", "parts": [{"text": "Hi"}]}]}
        try:
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code == 200:
                return {
                    "valid": True,
                    "status": "connected",
                    "model": model_name,
                    "message": f"Gemini API key verified successfully ({model_name})."
                }
            else:
                return {
                    "valid": False,
                    "status": "auth_failed",
                    "message": parse_gemini_error(resp.text, status_code=resp.status_code)
                }
        except Exception as ex:
            return {
                "valid": False,
                "status": "network_error",
                "message": parse_gemini_error(str(ex))
            }

    return {
        "valid": False,
        "status": "unknown_error",
        "message": "Unable to verify Gemini API key."
    }


def get_gemini_status(api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Comprehensive diagnostic reporting endpoint for GET /api/gemini/status.
    Reports key configuration, SDK status, active model, and connectivity.
    Never returns the full API key.
    """
    active_key, key_source = resolve_active_key(api_key)
    has_key = bool(active_key)

    result: Dict[str, Any] = {
        "key_configured": has_key,
        "key_source": key_source,
        "masked_key": mask_key(active_key) if has_key else None,
        "sdk_available": HAS_GOOGLE_GENAI,
        "sdk_version": SDK_VERSION,
        "configured_models": GEMINI_MODELS,
        "recommended_model": GEMINI_MODELS[0],
        "connection_tested": False,
        "connection_success": False,
        "message": "No key configured. Set GEMINI_API_KEY in .env or connect your key in the web interface."
    }

    if has_key:
        val_result = validate_api_key(active_key)
        result["connection_tested"] = True
        result["connection_success"] = val_result.get("valid", False)
        result["active_model"] = val_result.get("model", GEMINI_MODELS[0])
        result["message"] = val_result.get("message", "Key verified.")

    return result
