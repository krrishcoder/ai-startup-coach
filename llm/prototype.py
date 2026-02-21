from __future__ import annotations

from typing import Optional

from .openai_client import LLMUnavailable, OpenAIChat


def generate_prototype_html(
    *,
    idea: str,
    structured_plan: str,
    idea_type: str,
    llm: Optional[OpenAIChat] = None,
) -> str:
    idea = (idea or "").strip()
    idea_type = (idea_type or "").strip()

    system = "You generate a single self-contained HTML prototype file."
    user = f"""
Create a single HTML file prototype for this student startup idea.
Constraints:
- Single file HTML (no external assets)
- Clean, simple UI
- Must include: title, short description, 3-feature checklist, notes area
- Include minimal JS for local interactions (no backend)

Idea type: {idea_type}
Idea: {idea}

Structured plan:
{structured_plan}

Return ONLY the HTML.
""".strip()

    if llm is None:
        try:
            llm = OpenAIChat()
        except LLMUnavailable as e:
            raise RuntimeError(
                "Prototype generation is not working because OPENAI_API_KEY is not set. "
                "Add it to your .env (and optionally set OPENAI_MODEL)."
            ) from e

    html = llm.chat(system=system, user=user, temperature=0.2)
    if "<html" not in html.lower():
        raise RuntimeError(
            "Prototype generation is not working because the LLM did not return valid HTML."
        )

    return html
