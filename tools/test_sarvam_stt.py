#!/usr/bin/env python3
"""Quick CLI to test Sarvam speech-to-text WebSocket.

Usage:
  python tools/test_sarvam_stt.py path/to/sample.wav

Requirements:
  - Set `SARVAM_API_KEY` (and optionally `SARVAM_SPEECH_WS_URL`) in your environment or .env
  - `pip install websockets`

The script prints the raw JSON response and a simple transcript if available.
"""
from __future__ import annotations

import json
import os
import sys

from sarvam_speech_ws import speech_to_text_translate_blocking, SarvamWSUnavailable


def load_env_from_project():
    # best-effort load .env in project root
    p = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    os.environ.setdefault(k, v)
        except Exception:
            pass


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python tools/test_sarvam_stt.py path/to/sample.wav")
        return 2

    wav_path = sys.argv[1]
    if not os.path.exists(wav_path):
        print("File not found:", wav_path)
        return 2

    load_env_from_project()

    try:
        with open(wav_path, "rb") as f:
            audio = f.read()
    except Exception as e:
        print("Failed to read file:", e)
        return 3

    try:
        resp = speech_to_text_translate_blocking(audio_bytes=audio)
    except SarvamWSUnavailable as e:
        print("Sarvam WS unavailable:", e)
        print("Make sure SARVAM_API_KEY is set and 'websockets' is installed.")
        return 4
    except Exception as e:
        print("Error calling Sarvam WS:", e)
        return 5

    print("Raw response:")
    print(json.dumps(resp, indent=2, ensure_ascii=False))

    # Attempt to print a transcript if present
    data = resp.get("data") if isinstance(resp, dict) else None
    if data and isinstance(data, dict):
        transcript = data.get("transcript") or data.get("text") or data.get("result")
        if transcript:
            print("\nTranscript:\n", transcript)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
