import os
import requests
from typing import Dict, Any, Optional


# URL و اسم الموديل بتوع Ollama
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("LLM_MODEL", "phi3.5:3.8b-mini-instruct-q4_K_M")


def ollama_generate(
    prompt: str,
    model: Optional[str] = None,
    stream: bool = False,
    options: Optional[Dict[str, Any]] = None,
) -> str:
    """
    دالة بسيطة تبعت prompt لـ Ollama وترجع الـ response كنص.
    """
    if model is None:
        model = MODEL_NAME

    if options is None:
        options = {
            "num_predict": 256
            }

    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
        "options": options,
    }
    if options:
        payload["options"] = options

    url = f"{OLLAMA_URL}/api/generate"
    resp = requests.post(url, json=payload)
    resp.raise_for_status()

    if stream:
        full = ""
        for line in resp.iter_lines():
            if not line:
                continue
            full += line.decode("utf-8")
        return full
    else:
        data = resp.json()
        return data.get("response", "").strip()
