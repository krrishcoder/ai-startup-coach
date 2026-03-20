

# ai-startup-coach

> AI Startup Builder is a LangGraph-based assistant that turns student ideas into structured startup plans, validates them with real-time market insights, and generates a downloadable prototype, with multilingual support and persistent memory.

![License](https://img.shields.io/badge/license-Unknown-blue)
![Language](https://img.shields.io/badge/language-Python-informational)
![Stars](https://img.shields.io/badge/stars-1-yellow)

## 📌 Overview
This project is an AI-powered startup builder that assists students in turning their ideas into structured startup plans. It provides real-time market insights and generates a downloadable prototype. The system is multilingual and has persistent memory, allowing it to retain chat history in SQLite.

## 🚀 Features

* LangGraph-based assistant
* Turns student ideas into structured startup plans
* Validates startup plans with real-time market insights
* Generates downloadable prototypes
* Multilingual support
* Persistent memory for chat history in SQLite

## 🛠 Tech Stack

| Category | Details |
| --- | --- |
| Runtime | Python |
| Language | Python |

## 📦 Installation

```bash
git clone https://github.com/.../ai-startup-coach.git
cd ai-startup-coach
npm install
```

## ▶️ Usage

```bash
cd project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp ../.env.example .env
# fill keys in .env (optional)
python main.py
```

## 📁 Project Structure

```python
.gitignore
README.md
__init__.py
app.py
graph
  __init__.py
  workflow.py
llm
  __init__.py
  mentor.py
  openai_client.py
  prototype.py
  structure.py
main.py
memory
  __init__.py
  sqlite_memory.py
project
  memory
    assistant.db
    l.sql
prompts
  init_questions.txt
  language_select.txt
  market_research.txt
  mentor_feedback.txt
  mentor_questions.txt
  prototype.txt
  reddit_research.txt
  structure_idea.txt
requirements.txt
tools
  __init__.py
  sarvam_speech_ws.py
  sarvam_translate.py
  tavily_tool.py
  test_sarvam_stt.py
```

This project is a Python application with a directory structure that includes folders for the application, graph, language model, memory, project, prompts, requirements, and tools.

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a pull request

## 📄 License

This project is licensed under the MIT License.

---
FINAL CHECKLIST BEFORE OUTPUTTING:
  [ ] Tech Stack table — every row on its own line, none collapsed?
  [ ] File tree — inside a fenced code block, copied verbatim?
  [ ] No invented features or endpoints?
  [ ] No text outside the README document?
