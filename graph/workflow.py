from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

try:
    from langgraph.graph import END, StateGraph  # type: ignore
except Exception:  # pragma: no cover
    END = "__END__"  # type: ignore
    StateGraph = None  # type: ignore
from typing import TypedDict

from memory.sqlite_memory import Message, load_messages, save_messages
from tools.sarvam_translate import translate
from tools.tavily_tool import tavily_search

from llm.structure import structure_idea
from llm.mentor import generate_mentor_questions, mentor_feedback
from llm.prototype import generate_prototype_html


class GraphState(TypedDict, total=False):
    user_id: str
    conversation_id: str
    preferred_language: str
    idea: str
    class_level: int
    idea_type: str

    structured_plan: str
    market_info: str
    reddit_context: str
    research_context: str

    chat_history: list[dict[str, str]]
    mentor_questions: list[str]
    mentor_answers: list[str]
    mentor_feedback: str

    prototype_code: str
    prototype_path: str
    prototype_error: str

    _db_path: str
    _pending_messages: list[dict[str, str]]
    _io: Any  # optional interactive IO (CLI)


def _now_z() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _add_pending(state: GraphState, role: str, content: str) -> None:
    pending = state.get("_pending_messages") or []
    pending.append({"role": role, "content": content, "timestamp": _now_z()})
    state["_pending_messages"] = pending


def load_memory_node(state: GraphState) -> GraphState:
    db_path = state.get("_db_path") or os.getenv("ASSISTANT_DB_PATH", "project/memory/assistant.db")
    state["_db_path"] = db_path

    user_id = state.get("user_id", "").strip()
    conversation_id = state.get("conversation_id", "").strip()
    if not user_id or not conversation_id:
        state["chat_history"] = []
        return state

    msgs = load_messages(db_path, user_id=user_id, conversation_id=conversation_id, limit=80)
    state["chat_history"] = [{"role": m.role, "content": m.content} for m in msgs]
    return state


def collect_initial_questions_node(state: GraphState) -> GraphState:
    io = state.get("_io")
    if io is None:
        # Non-interactive: no initial questions
        state["initial_qa"] = []
        return state

    # Ask up to 3 simple questions to quickly understand the idea
    questions = [
        "Who will mainly use this app? (age/role)",
        "What is the single most important thing the app must do?",
        "How will you know if this idea is successful?",
    ]
    answers: list[tuple[str, str]] = []
    for q in questions:
        a = io.ask(f"{q}\n> ")
        a = (a or "").strip()
        if not a:
            continue
        answers.append((q, a))
        _add_pending(state, "user", f"{q} -> {a}")

    state["initial_qa"] = answers
    return state


def structure_idea_node(state: GraphState) -> GraphState:
    plan = structure_idea(
        idea=state.get("idea", ""),
        class_level=int(state.get("class_level") or 0),
        idea_type=state.get("idea_type", ""),
        initial_qa=state.get("initial_qa") or None,
        mentor_qa=None,
    )
    state["structured_plan"] = plan
    _add_pending(state, "assistant", plan)
    return state


def refine_structure_node(state: GraphState) -> GraphState:
    # Re-run structure generation using mentor Q&A to refine the plan
    initial = state.get("initial_qa") or None
    qs = state.get("mentor_questions") or []
    ans = state.get("mentor_answers") or []
    mentor_qa = [(q, a) for q, a in zip(qs, ans)] if qs and ans else None

    try:
        plan = structure_idea(
            idea=state.get("idea", ""),
            class_level=int(state.get("class_level") or 0),
            idea_type=state.get("idea_type", ""),
            initial_qa=initial,
            mentor_qa=mentor_qa,
        )
        state["structured_plan"] = plan
        _add_pending(state, "assistant", "Refined plan:\n" + plan)
    except Exception:
        # If refinement fails, leave original plan
        pass

    return state


def market_research_node(state: GraphState) -> GraphState:
    idea = state.get("idea", "").strip()
    if not idea:
        state["market_info"] = ""
        return state

    q = f"{idea} market competitors India"
    state["market_info"] = tavily_search(q, max_results=5)
    return state


def reddit_research_node(state: GraphState) -> GraphState:
    idea = state.get("idea", "").strip()
    if not idea:
        state["reddit_context"] = ""
        return state

    q = f"{idea} user problems discussion site:reddit.com"
    state["reddit_context"] = tavily_search(q, max_results=5)
    return state


def combine_research_node(state: GraphState) -> GraphState:
    market = (state.get("market_info") or "").strip()
    reddit = (state.get("reddit_context") or "").strip()
    combo = "\n\n".join([s for s in [market, reddit] if s])
    state["research_context"] = combo
    return state


def generate_mentor_questions_node(state: GraphState) -> GraphState:
    # Always start mentor Q&A fresh for this session
    state["mentor_questions"] = []
    state["mentor_answers"] = []
    state["mentor_q_loop_active"] = True
    state["mentor_q_session_context"] = {
        "structured_plan": state.get("structured_plan", ""),
        "research_context": state.get("research_context", ""),
        "class_level": int(state.get("class_level") or 0),
    }
    return state


def collect_answers_node(state: GraphState) -> GraphState:
    qs = state.get("mentor_questions") or []
    answers = state.get("mentor_answers") or []
    io = state.get("_io")
    if io is None:
        return state

    # If loop is inactive, skip
    if not state.get("mentor_q_loop_active", True):
        return state

    # Use only session context for generating questions
    session_ctx = state.get("mentor_q_session_context", {})
    structured_plan = session_ctx.get("structured_plan", "")
    research_context = session_ctx.get("research_context", "")
    class_level = session_ctx.get("class_level", 0)

    # Generate next question based on previous answers (but only for this session)
    next_idx = len(qs)
    if next_idx == 0:
        # First question
        q = generate_mentor_questions(
            structured_plan=structured_plan,
            research_context=research_context,
            class_level=class_level,
        )[0]
    else:
        # Adaptive question: use only this session's Q&A
        q = generate_mentor_questions(
            structured_plan=structured_plan,
            research_context=research_context,
            class_level=class_level,
            previous_questions=qs,
            previous_answers=answers,
        )[0]

    qs.append(q)
    state["mentor_questions"] = qs

    # Ask the question
    a = io.ask(f"Mentor Q{next_idx+1}: {q}\n> ")
    a = (a or "").strip()
    _add_pending(state, "user", a)
    answers.append(a)
    state["mentor_answers"] = answers

    # After each answer, generate a meaningful mentor response
    fb = mentor_feedback(
        structured_plan=structured_plan,
        research_context=research_context,
        questions=qs,
        answers=answers,
        class_level=class_level,
    )
    _add_pending(state, "assistant", fb)
    state["last_mentor_feedback"] = fb

    # Decide whether to continue the loop (stop after 3, or if mentor says so)
    # Loop continuation is handled by the CLI to allow refinement after each answer.
    # We do not prompt here; leave `mentor_q_loop_active` as-is so the caller can decide.

    return state


def mentor_feedback_node(state: GraphState) -> GraphState:
    qs = state.get("mentor_questions") or []
    answers = state.get("mentor_answers") or []
    # Only show summary feedback after Q&A loop is done
    if not qs or len(answers) < len(qs) or state.get("mentor_q_loop_active", True):
        state["mentor_feedback"] = ""
        return state

    fb = mentor_feedback(
        structured_plan=state.get("structured_plan", ""),
        research_context=state.get("research_context", ""),
        questions=qs,
        answers=answers,
        class_level=int(state.get("class_level") or 0),
    )
    state["mentor_feedback"] = fb
    _add_pending(state, "assistant", fb)
    return state


def translation_node(state: GraphState) -> GraphState:
    lang = (state.get("preferred_language") or "en").strip()
    if not lang or lang.lower() in {"en", "english"}:
        return state

    # Translate user-facing assistant outputs.
    if state.get("structured_plan"):
        state["structured_plan"] = translate(state["structured_plan"], target_language=lang)
    if state.get("mentor_feedback"):
        state["mentor_feedback"] = translate(state["mentor_feedback"], target_language=lang)

    qs = state.get("mentor_questions") or []
    if qs:
        joined = "\n".join([f"{i+1}. {q}" for i, q in enumerate(qs)])
        translated = translate(joined, target_language=lang)
        # Re-split as best effort
        new_qs = [line.strip().lstrip("0123456789").lstrip(".) ") for line in translated.splitlines() if line.strip()]
        state["mentor_questions"] = new_qs[: len(qs)] if new_qs else qs

    return state


def prototype_generator_node(state: GraphState) -> GraphState:
    try:
        html = generate_prototype_html(
            idea=state.get("idea", ""),
            structured_plan=state.get("structured_plan", ""),
            idea_type=state.get("idea_type", ""),
        )
        state["prototype_code"] = html
        state["prototype_error"] = ""
    except Exception as e:
        state["prototype_code"] = ""
        state["prototype_error"] = str(e) or "Prototype generation is not working."
    return state


def save_prototype_file_node(state: GraphState) -> GraphState:
    if not (state.get("prototype_code") or "").strip():
        msg = state.get("prototype_error") or "Prototype generation is not working."
        _add_pending(state, "assistant", msg)
        return state

    conv = (state.get("conversation_id") or "conv").strip() or "conv"
    out_dir = os.path.join(os.path.dirname(__file__), "..", "prototypes")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{conv}_app.html")

    code = state.get("prototype_code") or ""
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)

    state["prototype_path"] = path
    _add_pending(state, "assistant", f"Prototype generated: {path}")
    return state


def save_memory_node(state: GraphState) -> GraphState:
    db_path = state.get("_db_path") or os.getenv("ASSISTANT_DB_PATH", "project/memory/assistant.db")
    state["_db_path"] = db_path

    user_id = state.get("user_id", "").strip()
    conversation_id = state.get("conversation_id", "").strip()
    pending = state.get("_pending_messages") or []

    if not user_id or not conversation_id or not pending:
        return state

    msgs = [
        Message(
            user_id=user_id,
            conversation_id=conversation_id,
            role=m.get("role", "assistant"),
            content=m.get("content", ""),
            timestamp=m.get("timestamp", _now_z()),
        )
        for m in pending
        if (m.get("content") or "").strip()
    ]

    save_messages(db_path, msgs)
    state["_pending_messages"] = []
    return state


def build_workflow():
    # If LangGraph is available, build the real graph.
    if StateGraph is not None:
        g = StateGraph(GraphState)

        g.add_node("load_memory", load_memory_node)
        g.add_node("structure_idea", structure_idea_node)
        g.add_node("market_research", market_research_node)
        g.add_node("reddit_research", reddit_research_node)
        g.add_node("combine_research", combine_research_node)
        g.add_node("mentor_questions", generate_mentor_questions_node)
        g.add_node("collect_answers", collect_answers_node)
        g.add_node("mentor_feedback", mentor_feedback_node)
        g.add_node("refine_structure", refine_structure_node)
        g.add_node("translate", translation_node)
        g.add_node("prototype", prototype_generator_node)
        g.add_node("save_prototype", save_prototype_file_node)
        g.add_node("save_memory", save_memory_node)

        g.set_entry_point("load_memory")
        g.add_edge("load_memory", "structure_idea")
        g.add_edge("structure_idea", "market_research")
        g.add_edge("market_research", "reddit_research")
        g.add_edge("reddit_research", "combine_research")
        g.add_edge("combine_research", "mentor_questions")
        g.add_edge("mentor_questions", "collect_answers")
        g.add_edge("collect_answers", "mentor_feedback")
        g.add_edge("mentor_feedback", "refine_structure")
        g.add_edge("refine_structure", "translate")
        g.add_edge("translate", "prototype")
        g.add_edge("prototype", "save_prototype")
        g.add_edge("save_prototype", "save_memory")
        g.add_edge("save_memory", END)

        return g.compile()

    # Fallback: run nodes sequentially (keeps CLI usable without pip installs).
    class _SimpleWorkflow:
        def invoke(self, state: GraphState) -> GraphState:
            for fn in (
                load_memory_node,
                structure_idea_node,
                market_research_node,
                reddit_research_node,
                combine_research_node,
                generate_mentor_questions_node,
                collect_answers_node,
                mentor_feedback_node,
                translation_node,
                prototype_generator_node,
                save_prototype_file_node,
                save_memory_node,
            ):
                state = fn(state)
            return state

    return _SimpleWorkflow()
