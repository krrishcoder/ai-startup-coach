from __future__ import annotations

def _pick_language(io: CLIIO) -> str:
    options = [
        ("en", "English"),
        ("hi-IN", "Hindi"),
        ("ta-IN", "Tamil"),
        ("te-IN", "Telugu"),
        ("mr-IN", "Marathi"),
        ("bn-IN", "Bengali"),
        ("gu-IN", "Gujarati"),
        ("kn-IN", "Kannada"),
        ("ml-IN", "Malayalam"),
    ]
    print("Select preferred language:")
    for i, (_, name) in enumerate(options, start=1):
        print(f"{i}. {name}")
    raw = io.ask("> ").strip()
    try:
        idx = int(raw)
        if 1 <= idx <= len(options):
            return options[idx - 1][0]
    except Exception:
        pass
    codes = {code for code, _ in options}
    if raw in codes:
        return raw
    return "en"


class CLIIO:
    def ask(self, prompt: str) -> str:
        return input(prompt)



# CLIIO, _pick_language, _load_dotenv_fallback should be defined above or imported



def _load_dotenv_fallback(*paths: str) -> None:
    """Best-effort .env loader (only KEY=VALUE lines).
    Used when python-dotenv isn't installed.
    - Ignores comments/blank lines
    - Strips surrounding single/double quotes
    - Does not override existing environment variables
    """
    def _parse_line(line: str) -> tuple[str, str] | None:
        line = line.strip()
        if not line or line.startswith("#"):
            return None
        if line.lower().startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            return None
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            return None
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        return key, value
    for path in paths:
        if not path:
            continue
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                for raw in f:
                    parsed = _parse_line(raw)
                    if not parsed:
                        continue
                    key, value = parsed
                    os.environ.setdefault(key, value)
        except Exception:
            continue



import os
import uuid
import webbrowser
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    load_dotenv = None

from graph.workflow import (
    load_memory_node,
    collect_initial_questions_node,
    structure_idea_node,
    market_research_node,
    reddit_research_node,
    combine_research_node,
    generate_mentor_questions_node,
    collect_answers_node,
    mentor_feedback_node,
    refine_structure_node,
    translation_node,
    prototype_generator_node,
    save_prototype_file_node,
    save_memory_node,
)
from memory.sqlite_memory import save_message

# CLIIO, _pick_language, _load_dotenv_fallback should be defined above or imported

def main() -> None:
    if load_dotenv is not None:
        load_dotenv()
    else:
        cwd_env = os.path.join(os.getcwd(), ".env")
        root_env = os.path.abspath(os.path.join(os.getcwd(), "..", ".env"))
        _load_dotenv_fallback(cwd_env, root_env)
    io = CLIIO()

    db_path = os.getenv("ASSISTANT_DB_PATH", "project/memory/assistant.db")

    print("AI Startup Builder (CLI)")
    user_id = io.ask("User ID: ").strip() or "student"
    conversation_id = io.ask("Conversation ID (blank = new): ").strip()
    if not conversation_id:
        conversation_id = f"conv_{uuid.uuid4().hex[:8]}"
    preferred_language = _pick_language(io)

    # Initial simple questions
    idea = io.ask("Describe your startup idea in 1-2 lines: ").strip()
    class_level_raw = io.ask("Your class (6-12): ").strip()
    try:
        class_level = int(class_level_raw)
    except Exception:
        class_level = 8
    idea_type = io.ask("Idea type (App/Website, AI Tool, Marketplace): ").strip() or "App/Website"

    save_message(
        db_path,
        user_id=user_id,
        conversation_id=conversation_id,
        role="user",
        content=f"Idea: {idea}\nClass: {class_level}\nType: {idea_type}\nLanguage: {preferred_language}",
    )

    # Build initial state and run the preparation nodes up to combined research
    state = {
        "user_id": user_id,
        "conversation_id": conversation_id,
        "preferred_language": preferred_language,
        "idea": idea,
        "class_level": class_level,
        "idea_type": idea_type,
        "_db_path": db_path,
        "_io": io,
        "_pending_messages": [],
    }
    # Preparation nodes
    state = load_memory_node(state)
    state = collect_initial_questions_node(state)
    state = structure_idea_node(state)
    state = market_research_node(state)
    state = reddit_research_node(state)
    state = combine_research_node(state)

    print("\n--- Structured Plan ---\n")
    print(state.get("structured_plan", ""))
    input("\nPress Enter to continue to mentor questions...")

    # Initialize mentor Q&A session
    state = generate_mentor_questions_node(state)

    # Show research context if available
    market_info = state.get("market_info", "")
    reddit_context = state.get("reddit_context", "")
    if market_info or reddit_context:
        print("\n[Market/Reddit Research Context]")
        if market_info:
            print("Market info:\n" + market_info)
        if reddit_context:
            print("Reddit context:\n" + reddit_context)

    # Iterative mentor Q&A -> refine -> ask user whether to continue
    while state.get("mentor_q_loop_active", True):
        state = collect_answers_node(state)

        # After each answer, refine the structured plan using mentor Q&A
        state = refine_structure_node(state)
        print("\n--- Refined Structured Plan ---\n")
        print(state.get("structured_plan", ""))

        cont = io.ask("Continue mentor Q&A and refine again? (y/n): ").strip().lower()
        if cont != "y":
            state["mentor_q_loop_active"] = False

    # After loop ends, generate final mentor feedback summary
    state = mentor_feedback_node(state)
    if state.get("mentor_feedback"):
        print("\n--- Mentor Feedback ---\n")
        print(state.get("mentor_feedback", ""))

    # Translation, prototype generation and saving
    state = translation_node(state)
    state = prototype_generator_node(state)
    state = save_prototype_file_node(state)

    proto_path = state.get("prototype_path", "")
    if proto_path:
        print("\nPrototype generated successfully.")
        print(f"File saved at:\n{proto_path}")
        open_now = io.ask("Open in browser now? (y/n): ").strip().lower() == "y"
        if open_now:
            webbrowser.open(f"file://{proto_path}")

    # Persist conversation to memory
    state = save_memory_node(state)

    print(f"\nConversation ID: {conversation_id}")


if __name__ == "__main__":
    main()
