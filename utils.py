import os
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple

from dotenv import load_dotenv
import requests

load_dotenv()

# =============== OLLAMA CONFIG ===============

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

# Embedding config (embedding.py uses these)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")
EMBEDDING_URL = os.getenv("EMBEDDING_URL", f"{OLLAMA_HOST}/api/embed")

# Directories
PDF_DIR = os.getenv("PDF_DIR", "data/pdfs")
INDEX_DIR = os.getenv("INDEX_DIR", "data/index")


# =============== CHAT COMPLETION (OLLAMA) ===============

def call_lmstudio(system_prompt: str, user_prompt: str, temperature: float = 0.2,
                  max_tokens: int = 1024) -> str:
    """
    Call Ollama's /api/chat endpoint.
    (We keep the function name call_lmstudio so rag.py works without changes.)
    """

    url = f"{OLLAMA_HOST}/api/chat"

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        "temperature": temperature,
        "stream": False,
    }

    resp = requests.post(url, json=payload, timeout=3000)
    resp.raise_for_status()

    data = resp.json()

    # Expected format:
    # {
    #   "message": { "role": "assistant", "content": "..." }
    # }
    if "message" in data and "content" in data["message"]:
        return data["message"]["content"]

    raise RuntimeError(f"Unexpected response from Ollama: {data}")
