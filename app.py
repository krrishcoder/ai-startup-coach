from __future__ import annotations

import asyncio
import json
import os
import uuid
from typing import Any, Dict, Optional


# Best-effort .env loader so running `uvicorn app:app` picks up project/.env
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except Exception:
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" not in line:
                        continue
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    if key and val:
                        os.environ.setdefault(key, val)
        except Exception:
            pass

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
import base64
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from memory.sqlite_memory import save_message, load_messages
from llm.openai_client import OpenAIChat, LLMUnavailable
from llm.prototype import generate_prototype_html
from llm.structure import structure_idea
from llm.mentor import generate_mentor_questions, mentor_feedback
from tools.tavily_tool import tavily_search
from graph.workflow import (
    load_memory_node,
    collect_initial_questions_node,
    structure_idea_node,
    market_research_node,
    reddit_research_node,
    combine_research_node,
    generate_mentor_questions_node,
    refine_structure_node,
    mentor_feedback_node,
)

try:
    from tools.sarvam_speech_ws import (
        speech_to_text_translate_blocking,
        SarvamWSUnavailable,
    )
except Exception:  # pragma: no cover - optional dependency
    speech_to_text_translate_blocking = None  # type: ignore
    SarvamWSUnavailable = Exception  # type: ignore

app = FastAPI()

# Allow the Vite dev server and local testing origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


class ChatRequest(BaseModel):
    conversation_id: Optional[str] = None
    user_id: Optional[str] = "user"
    user_message: str


@app.post("/api/chat")
async def chat(req: ChatRequest) -> Dict[str, Any]:
    conv = req.conversation_id or f"conv_{uuid.uuid4().hex[:8]}"
    user_id = req.user_id or "user"
    db_path = os.getenv("ASSISTANT_DB_PATH", "project/memory/assistant.db")

    # save user message to memory
    try:
        save_message(db_path, user_id=user_id, conversation_id=conv, role="user", content=req.user_message)
    except Exception:
        pass

    # Attempt to call LLM
    try:
        llm = OpenAIChat()
        system = "You are a helpful startup mentor assistant. Keep responses concise and actionable."
        assistant_text = llm.chat(system=system, user=req.user_message)
    except LLMUnavailable:
        assistant_text = "(Assistant unavailable - OPENAI_API_KEY not set)"
    except Exception as e:
        assistant_text = f"(Assistant error: {e})"

    # persist assistant reply
    try:
        save_message(db_path, user_id=user_id, conversation_id=conv, role="assistant", content=assistant_text)
    except Exception:
        pass

    return {"conversation_id": conv, "reply": assistant_text}


@app.get("/api/conversation/{conversation_id}")
async def get_conversation(conversation_id: str, user_id: Optional[str] = "user") -> Dict[str, Any]:
    db_path = os.getenv("ASSISTANT_DB_PATH", "project/memory/assistant.db")
    try:
        msgs = load_messages(db_path, user_id=user_id or "user", conversation_id=conversation_id, limit=500)
    except Exception:
        msgs = []
    return {"messages": [m.__dict__ for m in msgs]}


class PrototypeRequest(BaseModel):
    idea: str
    structured_plan: str
    idea_type: Optional[str] = "App/Website"


@app.post("/api/prototype")
async def create_prototype(req: PrototypeRequest) -> Dict[str, Any]:
    try:
        html = generate_prototype_html(idea=req.idea, structured_plan=req.structured_plan, idea_type=req.idea_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "prototypes"))
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{uuid.uuid4().hex[:8]}_app.html"
    path = os.path.join(out_dir, fname)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)

    return {"path": path}


class InitRequest(BaseModel):
    idea: str
    class_level: int = 8
    idea_type: Optional[str] = "App/Website"
    user_id: Optional[str] = "user"
    conversation_id: Optional[str] = None


@app.post("/api/initialize")
async def initialize(req: InitRequest) -> Dict[str, Any]:
    """Run initial structuring and research and return structured plan + first mentor question."""
    conv = req.conversation_id or f"conv_{uuid.uuid4().hex[:8]}"
    db_path = os.getenv("ASSISTANT_DB_PATH", "project/memory/assistant.db")

    # Save initial idea as user message
    try:
        save_message(db_path, user_id=req.user_id or "user", conversation_id=conv, role="user", content=req.idea)
    except Exception:
        pass

    # Build a workflow state and run the same nodes used by the CLI to produce plan + research + first mentor Q
    state = {
        "user_id": req.user_id or "user",
        "conversation_id": conv,
        "preferred_language": os.getenv("PREFERRED_LANGUAGE", "en"),
        "idea": req.idea,
        "class_level": int(req.class_level or 0),
        "idea_type": req.idea_type or "",
        "_db_path": db_path,
        "_io": None,
        "_pending_messages": [],
    }

    # Run preparation nodes (no interactive IO)
    state = load_memory_node(state)
    state = collect_initial_questions_node(state)
    state = structure_idea_node(state)
    state = market_research_node(state)
    state = reddit_research_node(state)
    state = combine_research_node(state)
    state = generate_mentor_questions_node(state)

    plan = state.get("structured_plan", "")
    market_info = state.get("market_info", "")
    reddit_context = state.get("reddit_context", "")
    research_context = state.get("research_context", "")
    mentor_questions = state.get("mentor_questions", []) or []
    first_q = mentor_questions[0] if mentor_questions else None

    # Market and reddit research (best-effort)
    market_q = f"{req.idea} market competitors India"
    reddit_q = f"{req.idea} user problems discussion site:reddit.com"
    market_info = tavily_search(market_q, max_results=5)
    reddit_context = tavily_search(reddit_q, max_results=5)
    research_context = "\n\n".join([s for s in [market_info, reddit_context] if s])

    # First mentor question
    questions = generate_mentor_questions(structured_plan=plan, research_context=research_context, class_level=int(req.class_level or 0))
    first_q = questions[0] if questions else "What specific pain points do your target users face?"

    return {
        "conversation_id": conv,
        "structured_plan": plan,
        "market_info": market_info,
        "reddit_context": reddit_context,
        "research_context": research_context,
        "mentor_question": first_q,
    }


class MentorAnswerRequest(BaseModel):
    idea: str
    class_level: int = 8
    idea_type: Optional[str] = "App/Website"
    research_context: Optional[str] = ""
    previous_questions: Optional[list[str]] = None
    previous_answers: Optional[list[str]] = None


@app.post("/api/mentor/answer")
async def mentor_answer(req: MentorAnswerRequest) -> Dict[str, Any]:
    """Accept previous Q&A, return next question, refined plan, and mentor feedback."""
    prev_qs = req.previous_questions or []
    prev_as = req.previous_answers or []

    # Build a workflow-like state and call refinement + feedback nodes to match CLI behavior
    state = {
        "user_id": "user",
        "conversation_id": None,
        "preferred_language": os.getenv("PREFERRED_LANGUAGE", "en"),
        "idea": req.idea,
        "class_level": int(req.class_level or 0),
        "idea_type": req.idea_type or "",
        "research_context": req.research_context or "",
        "mentor_questions": prev_qs,
        "mentor_answers": prev_as,
        "_db_path": os.getenv("ASSISTANT_DB_PATH", "project/memory/assistant.db"),
        "_pending_messages": [],
    }

    # Use workflow nodes to refine plan and generate feedback
    state = refine_structure_node(state)
    state = mentor_feedback_node(state)

    refined = state.get("structured_plan", "")
    fb = state.get("mentor_feedback", "")

    # Compute next question adaptively using the same LLM helper
    next_qs = generate_mentor_questions(
        structured_plan=refined,
        research_context=state.get("research_context", "") or "",
        class_level=int(req.class_level or 0),
        previous_questions=prev_qs,
        previous_answers=prev_as,
    )
    next_q = next_qs[0] if next_qs else None

    return {"next_question": next_q, "refined_plan": refined, "mentor_feedback": fb}


@app.post("/api/stt")
async def stt_one_shot(request: Request) -> Dict[str, Any]:
    """Accepts either JSON {"audio_b64": "..."} or raw binary body (wav) and returns STT result.

    This avoids the `UploadFile`/multipart dependency so the server can start without
    `python-multipart` installed. Clients may POST JSON with base64 audio or send
    the wav bytes as the request body with `Content-Type: audio/wav`.
    """
    if speech_to_text_translate_blocking is None:
        raise HTTPException(status_code=501, detail="STT backend not available on server (missing dependency)")

    ctype = (request.headers.get("content-type") or "").lower()
    try:
        if "application/json" in ctype:
            body = await request.json()
            b64 = body.get("audio_b64")
            if not b64:
                raise HTTPException(status_code=400, detail="JSON must include 'audio_b64' base64 string")
            data = base64.b64decode(b64)
        else:
            # read raw body bytes
            data = await request.body()

        resp = speech_to_text_translate_blocking(audio_bytes=data)
    except SarvamWSUnavailable as e:
        raise HTTPException(status_code=502, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"raw": resp}


@app.websocket("/api/ws/sarvam")
async def sarvam_ws_proxy(ws: WebSocket):
    """Proxy: accept client WS then bridge to Sarvam WS, adding api-subscription-key header.

    Client messages are forwarded to Sarvam and vice-versa. This keeps SARVAM_API_KEY on server.
    """
    await ws.accept()

    # Import websockets lazily
    try:
        import websockets
    except Exception:
        await ws.send_text(json.dumps({"error": "Server missing websockets package"}))
        await ws.close()
        return

    sarvam_url = os.getenv("SARVAM_SPEECH_WS_URL") or "wss://api.sarvam.ai/speech-to-text-translate/ws"
    api_key = os.getenv("SARVAM_API_KEY", "").strip()
    if not api_key:
        await ws.send_text(json.dumps({"error": "SARVAM_API_KEY not configured on server"}))
        await ws.close()
        return

    # Bridge tasks
    async def forward_to_sarvam(client_ws: WebSocket, target_uri: str):
        extra = [("api-subscription-key", api_key)]
        async with websockets.connect(target_uri, extra_headers=dict(extra)) as sarvam_ws:
            async def from_client():
                try:
                    while True:
                        msg = await client_ws.receive_text()
                        await sarvam_ws.send(msg)
                except WebSocketDisconnect:
                    await sarvam_ws.close()
                except Exception:
                    try:
                        await sarvam_ws.close()
                    except Exception:
                        pass

            async def from_sarvam():
                try:
                    async for m in sarvam_ws:
                        if isinstance(m, bytes):
                            await client_ws.send_text(m.decode("utf-8", errors="replace"))
                        else:
                            await client_ws.send_text(m)
                except Exception:
                    try:
                        await client_ws.close()
                    except Exception:
                        pass

            await asyncio.gather(from_client(), from_sarvam())

    try:
        await forward_to_sarvam(ws, sarvam_url)
    except Exception as e:
        try:
            await ws.send_text(json.dumps({"error": str(e)}))
        except Exception:
            pass
        try:
            await ws.close()
        except Exception:
            pass
