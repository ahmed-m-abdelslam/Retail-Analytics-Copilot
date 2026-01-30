# Retail Analytics Copilot (DSPy + LangGraph)

### Overview
- Build a local, free AI agent that answers retail analytics questions by combining:
    - RAG over local docs (docs/)
    - SQL over a local SQLite DB (Northwind)
- Produce typed, auditable answers with citations.
- Use DSPy to optimize at least one component.
- No paid APIs or external calls at inference time.
- Estimated time: 2–3 focused hours
- Runs on: normal PC (CPU ok), 16GB RAM recommended
- Local model constraint: Phi-3.5-mini-instruct via Ollama (or llama.cpp GGUF)