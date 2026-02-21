from __future__ import annotations

from typing import Optional
from typing import Sequence, Tuple

from .openai_client import LLMUnavailable, OpenAIChat

def _format_qa_list(qa: Optional[Sequence[Tuple[str, str]]]) -> str:
    if not qa:
        return ""
    lines: list[str] = []
    for q, a in qa:
        q = (q or "").strip()
        a = (a or "").strip()
        if q or a:
            lines.append(f"Q: {q}\nA: {a}")
    return "\n\n".join(lines)


def structure_idea(
    *,
    idea: str,
    class_level: int,
    idea_type: str,
    initial_qa: Optional[Sequence[Tuple[str, str]]] = None,
    mentor_qa: Optional[Sequence[Tuple[str, str]]] = None,
    llm: Optional[OpenAIChat] = None,
) -> str:
    idea = (idea or "").strip()
    idea_type = (idea_type or "").strip()

    if not idea:
        return "(No idea provided.)"

    difficulty = "simple" if class_level <= 8 else "detailed"

    system = (
        "You are a startup mentor for Indian school students. "
        "Output must be clear, structured, and practical."
    )

    init_ctx = _format_qa_list(initial_qa)
    mentor_ctx = _format_qa_list(mentor_qa)

    user = f"""
Student class: {class_level}
Idea type: {idea_type}
Writing level: {difficulty}

Startup idea: {idea}

Initial notes (from quick questions):
{init_ctx}

Mentor Q&A so far:
{mentor_ctx}

Create a structured plan with these headings:
1) Problem Statement
2) Target Users
3) Key Features (3 bullets)
4) Revenue Model
5) 5-Day Validation Plan (Day 1..Day 5)

Keep it short and student-friendly.
""".strip()

    if llm is None:
        try:
            llm = OpenAIChat()
        except LLMUnavailable:
            llm = None
    if llm is None:
        base = (
            "1) Problem Statement\n"
            f"- {idea}\n\n"
            "2) Target Users\n"
            "- Students / parents / teachers (choose a clear group)\n\n"
            "3) Key Features\n"
            "- Feature 1: Simple onboarding\n"
            "- Feature 2: Core workflow to solve the problem\n"
            "- Feature 3: Basic tracking / reminders\n\n"
            "4) Revenue Model\n"
            "- Free + premium OR school subscription\n\n"
            "5) 5-Day Validation Plan\n"
            "- Day 1: Talk to 5 users\n"
            "- Day 2: Define 3 must-have features\n"
            "- Day 3: Paper prototype / Figma\n"
            "- Day 4: Build HTML prototype\n"
            "- Day 5: Test with users and improve\n"
        )
        if mentor_ctx:
            base += "\n(Refined after mentor Q&A)\n"
        return base

    return llm.chat(system=system, user=user, temperature=0.2)
