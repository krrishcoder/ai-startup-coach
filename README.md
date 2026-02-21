# AI Startup Builder (CLI)

## Run

```bash
cd project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp ../.env.example .env
# fill keys in .env (optional)
python main.py
```

## Notes
- Works without API keys (uses safe fallback responses).
- Prototypes are written to `project/prototypes/`.
- Chat history persists in SQLite (`project/memory/assistant.db`).
# ai-startup-coach
