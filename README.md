# rag-agent-system

Enterprise knowledge base RAG Agent system MVP.

This is the first runnable prototype. It does not connect to a vector database or LLM yet. It provides a FastAPI service with:

- `GET /` demo landing response
- `GET /health` health check
- `POST /api/chat` mock chat response
- `GET /docs` automatic API documentation

## Run locally

```powershell
cd "D:\Code\codex\AI lab\rag-agent-system"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open:

```text
http://localhost:8000
http://localhost:8000/docs
```

## Test chat API

```powershell
Invoke-RestMethod `
  -Uri "http://localhost:8000/api/chat" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"question":"这个系统现在支持什么？"}'
```

