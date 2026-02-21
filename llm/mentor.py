from __future__ import annotations

from typing import Optional

from .openai_client import LLMUnavailable, OpenAIChat


def generate_mentor_questions(
    *,
    structured_plan: str,
    research_context: str,
    class_level: int,
    previous_questions: list[str] = None,
    previous_answers: list[str] = None,
    llm: Optional[OpenAIChat] = None,
) -> list[str]:
    system = "You are a strict but kind startup mentor. Ask one intelligent follow-up question at a time, referencing previous answers and research context."
    previous_qa = ""
    if previous_questions and previous_answers:
        for i, (q, a) in enumerate(zip(previous_questions, previous_answers), start=1):
            previous_qa += f"Q{i}: {q}\nA{i}: {a}\n"
    user = f"""
Student class: {class_level}

Structured plan:
{structured_plan}

Research context (market + reddit):
{research_context}

Previous Q&A:
{previous_qa}

Ask the next mentor question. Output only the question, no extra text.
""".strip()

    if llm is None:
        try:
            llm = OpenAIChat()
        except LLMUnavailable:
            llm = None

    if llm is None:
        # Fallback: basic adaptive logic
        if previous_questions and len(previous_questions) > 0:
            return ["What will you do next to validate your idea?"]
        return ["What specific pain points do your target users face?"]

    text = llm.chat(system=system, user=user, temperature=0.2)
    question = text.strip().splitlines()[0] if text.strip() else "What will you do next?"
    return [question]


def mentor_feedback(
    *,
    structured_plan: str,
    research_context: str,
    questions: list[str],
    answers: list[str],
    class_level: int,
    llm: Optional[OpenAIChat] = None,
) -> str:
    system = "You are a startup mentor. Give actionable feedback and next steps."

    qa = "\n".join(
        f"Q{i+1}: {q}\nA{i+1}: {a}" for i, (q, a) in enumerate(zip(questions, answers))
    )

    user = f"""
Student class: {class_level}

Structured plan:
{structured_plan}

Research context (market + reddit):
{research_context}

Mentor Q&A:
{qa}

Give feedback with:
- 3 strengths
- 3 risks/unknowns
- 5 next actions for this week
Keep it student-friendly.
""".strip()

    if llm is None:
        try:
            llm = OpenAIChat()
        except LLMUnavailable:
            llm = None

    if llm is None:
        return (
            "Strengths:\n"
            "- Clear problem and audience\n"
            "- Simple feature set\n"
            "- Practical weekly plan\n\n"
            "Risks / Unknowns:\n"
            "- Need proof people will use it\n"
            "- Competitors might already exist\n"
            "- Pricing/payment unclear\n\n"
            "Next actions (this week):\n"
            "1) Interview 5 target users\n"
            "2) Write the top 3 pain points you heard\n"
            "3) Build the HTML prototype\n"
            "4) Test prototype with 5 users\n"
            "5) Update plan based on feedback\n"
        )

    return llm.chat(system=system, user=user, temperature=0.2)
