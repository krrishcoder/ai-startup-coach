from __future__ import annotations

import os
from typing import Any


def tavily_search(query: str, *, max_results: int = 5) -> str:
    """Return a compact text summary of Tavily results.

    If `TAVILY_API_KEY` isn't configured or the Tavily client isn't installed,
    returns an empty string.
    """

    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        return ""

    try:
        from tavily import TavilyClient  # type: ignore
    except Exception:
        return ""

    try:
        client = TavilyClient(api_key=api_key)
        result: Any = client.search(query=query, max_results=max_results)
        items = result.get("results", []) if isinstance(result, dict) else []

        lines: list[str] = []
        for i, item in enumerate(items[:max_results], start=1):
            title = (item.get("title") or "").strip()
            url = (item.get("url") or "").strip()
            snippet = (item.get("content") or item.get("snippet") or "").strip()
            snippet = snippet.replace("\n", " ").strip()
            if snippet:
                snippet = snippet[:240] + ("…" if len(snippet) > 240 else "")
            lines.append(f"{i}. {title}\n{url}\n{snippet}".strip())

        return "\n\n".join([l for l in lines if l])
    except Exception:
        return ""
