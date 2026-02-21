from __future__ import annotations

import os
from typing import Optional

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore


def translate(text: str, *, target_language: str) -> str:
    """Translate `text` into `target_language`.

    This is intentionally defensive because Sarvam API details can vary by account.
    Configure:
      - `SARVAM_API_KEY`
      - `SARVAM_TRANSLATE_URL`

    If not configured (or on error), returns the original text.
    """

    if not text.strip():
        return text

    if target_language.lower() in {"en", "english"}:
        return text

    api_key = os.getenv("SARVAM_API_KEY", "").strip()
    url = os.getenv("SARVAM_TRANSLATE_URL", "").strip()
    if not api_key or not url:
        return text

    if requests is None:
        return text

    try:
        resp = requests.post(
            url,
            headers={
                # Sarvam expects this header name per their docs/examples.
                "api-subscription-key": api_key,
                "Content-Type": "application/json",
            },
            json={
                "input": text,
                "source_language_code": "auto",
                "target_language_code": target_language,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        # Try common shapes
        if isinstance(data, dict):
            for key in (
                "translated_text",
                "translation",
                "output",
                "result",
                "translatedText",
            ):
                val = data.get(key)
                if isinstance(val, str) and val.strip():
                    return val
            # Sometimes: { data: { translated_text: "..." } }
            inner = data.get("data")
            if isinstance(inner, dict):
                val = inner.get("translated_text")
                if isinstance(val, str) and val.strip():
                    return val

            # Sometimes: { output: [ { translated_text: "..." } ] }
            output = data.get("output")
            if isinstance(output, list) and output:
                first = output[0]
                if isinstance(first, dict):
                    val = first.get("translated_text") or first.get("translatedText")
                    if isinstance(val, str) and val.strip():
                        return val

        return text
    except Exception:
        return text
