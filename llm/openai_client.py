from __future__ import annotations

import os
import json
import urllib.request
import urllib.error
from typing import Optional


class LLMUnavailable(Exception):
    pass


class OpenAIChat:
    def __init__(self, *, model: Optional[str] = None):
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

        if not self.api_key:
            raise LLMUnavailable("OPENAI_API_KEY is not set")

        # Prefer official SDK if available, otherwise fall back to raw HTTPS.
        self._sdk_client = None
        try:
            from openai import OpenAI  # type: ignore

            self._sdk_client = OpenAI(api_key=self.api_key)
        except Exception:
            self._sdk_client = None

    def chat(self, *, system: str, user: str, temperature: float = 0.2) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        if self._sdk_client is not None:
            resp = self._sdk_client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                messages=messages,
            )
            msg = resp.choices[0].message
            content = getattr(msg, "content", None)
            return (content or "").strip()

        # Raw REST fallback (no external deps)
        url = "https://api.openai.com/v1/chat/completions"
        body = {
            "model": self.model,
            "temperature": temperature,
            "messages": messages,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read().decode("utf-8")
            data = json.loads(raw)
            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            return (content or "").strip()
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", errors="replace")
            except Exception:
                detail = ""
            raise RuntimeError(f"OpenAI API HTTP error: {e.code} {e.reason} {detail}".strip()) from e
        except Exception as e:
            raise RuntimeError("OpenAI API request failed") from e
