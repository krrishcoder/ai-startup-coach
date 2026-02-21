from __future__ import annotations

import asyncio
import base64
import json
import os
from typing import Any, Optional


class SarvamWSUnavailable(Exception):
    pass


def _get_ws_url() -> str:
    return os.getenv("SARVAM_SPEECH_WS_URL", "wss://api.sarvam.ai/speech-to-text-translate/ws").strip()


def _get_api_key() -> str:
    return os.getenv("SARVAM_API_KEY", "").strip()


async def speech_to_text_translate(
    *,
    audio_bytes: bytes,
    sample_rate: int = 16000,
    encoding: str = "audio/wav",
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    """Send audio to Sarvam STT Translate WebSocket and return the first data message.

    Expects Sarvam API docs shape:
      request: {"audio": {"data": "<base64>", "sample_rate": "16000", "encoding": "audio/wav"}}
      response: {"type": "data", "data": {"transcript": "...", ...}}

    Notes:
    - This implementation sends a single message (one-shot). For true streaming,
      you typically send multiple smaller chunks and aggregate partial results.
    - If `websockets` isn't installed or SARVAM_API_KEY missing, raises.
    """

    api_key = _get_api_key()
    if not api_key:
        raise SarvamWSUnavailable("SARVAM_API_KEY is not set")

    try:
        import websockets  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SarvamWSUnavailable("Install dependency: pip install websockets") from e

    url = _get_ws_url()

    payload = {
        "audio": {
            "data": base64.b64encode(audio_bytes).decode("utf-8"),
            "sample_rate": str(sample_rate),
            "encoding": encoding,
        }
    }

    # Header name is based on Sarvam docs style.
    extra_headers = {
        "api-subscription-key": api_key,
    }

    async with websockets.connect(url, extra_headers=extra_headers) as ws:
        await ws.send(json.dumps(payload))

        async def _recv_loop() -> dict[str, Any]:
            while True:
                msg = await ws.recv()
                if isinstance(msg, bytes):
                    msg = msg.decode("utf-8", errors="replace")
                data = json.loads(msg)
                if isinstance(data, dict) and data.get("type") == "data":
                    return data

        return await asyncio.wait_for(_recv_loop(), timeout=timeout_s)


def speech_to_text_translate_blocking(
    *,
    audio_bytes: bytes,
    sample_rate: int = 16000,
    encoding: str = "audio/wav",
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    """Sync wrapper for quick scripts."""

    return asyncio.run(
        speech_to_text_translate(
            audio_bytes=audio_bytes,
            sample_rate=sample_rate,
            encoding=encoding,
            timeout_s=timeout_s,
        )
    )
