"""
config.py - Shared runtime configuration helpers.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import dotenv_values, load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOTENV_PATH = PROJECT_ROOT / ".env"
BOM_PREFIX = "\ufeff"


def _load_api_key_from_dotenv_file() -> Optional[str]:
    """Reads the API key directly from .env, including UTF-8 BOM-tolerant files."""

    if not DOTENV_PATH.exists():
        return None

    raw_values = dotenv_values(DOTENV_PATH, encoding="utf-8-sig")
    for key_name, value in raw_values.items():
        normalized_key = key_name.lstrip(BOM_PREFIX)
        if normalized_key == "OPENAI_API_KEY" and value:
            os.environ["OPENAI_API_KEY"] = str(value)
            return str(value)
    return None


def resolve_openai_api_key() -> Optional[str]:
    """Returns the OpenAI API key from env or Streamlit secrets."""

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key

    bom_api_key = os.getenv(f"{BOM_PREFIX}OPENAI_API_KEY")
    if bom_api_key:
        os.environ["OPENAI_API_KEY"] = bom_api_key
        return bom_api_key

    dotenv_api_key = _load_api_key_from_dotenv_file()
    if dotenv_api_key:
        return dotenv_api_key

    try:
        import streamlit as st
    except Exception:
        return None

    candidate_keys = ("OPENAI_API_KEY", "openai_api_key")
    for key_name in candidate_keys:
        try:
            value = st.secrets.get(key_name)
        except Exception:
            value = None
        if value:
            os.environ["OPENAI_API_KEY"] = str(value)
            return str(value)

    try:
        openai_section = st.secrets.get("openai")
    except Exception:
        openai_section = None

    if openai_section:
        for key_name in ("api_key", "OPENAI_API_KEY", "openai_api_key"):
            value = openai_section.get(key_name)
            if value:
                os.environ["OPENAI_API_KEY"] = str(value)
                return str(value)

    return None


def openai_api_key_help_text() -> str:
    """Returns a user-facing setup message for missing API keys."""

    return (
        "OpenAI API key not configured. Add `OPENAI_API_KEY` to your local `.env` "
        "file, or set `OPENAI_API_KEY` in Streamlit secrets for deployment."
    )
