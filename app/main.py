from datetime import datetime, timezone
from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


app = FastAPI(
    title="RAG Agent System MVP",
    description="A minimal local prototype for the Yunhao AI Lab RAG Agent project.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    version: str
    timestamp: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class Source(BaseModel):
    title: str
    content: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    mode: Literal["mock"]


@app.get("/")
def read_root() -> dict[str, str]:
    return {
        "name": "RAG Agent System MVP",
        "status": "running",
        "docs": "http://localhost:8000/docs",
        "health": "http://localhost:8000/health",
    }


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="rag-agent-system",
        version="0.1.0",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    return ChatResponse(
        answer=(
            "这是 rag-agent-system 的最小原型回答。当前版本还没有接入真实 LLM、"
            "向量数据库或文档解析流程，但已经完成了 FastAPI 服务、健康检查、"
            "CORS 配置和 mock 问答接口。你刚才的问题是："
            f"{request.question}"
        ),
        sources=[
            Source(
                title="MVP Scope",
                content="当前阶段先验证本地服务、API 结构和个人主页 Demo 跳转。",
            ),
            Source(
                title="Next Step",
                content="下一步可以加入文档上传、文本切分、向量检索和真实 RAG 问答。",
            ),
        ],
        mode="mock",
    )

