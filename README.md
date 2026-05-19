# rag-agent-system

企业知识库 RAG Agent 系统最小原型。

这个项目用于展示大模型应用开发中的 RAG、Agent、文档处理、向量检索和后端工程能力。当前版本是第一阶段的本地可运行原型，暂时还没有接入真实 LLM、向量数据库和文档解析流程。

## 当前功能

- FastAPI 后端服务
- 健康检查接口：`GET /health`
- 服务根路由：`GET /`
- Mock 问答接口：`POST /api/chat`
- Swagger API 文档：`GET /docs`
- 支持个人主页 `http://localhost:3000` 跨域访问
- 支持前端 Demo 页面联调

## 技术栈

- Python
- FastAPI
- Pydantic
- Uvicorn

后续计划接入：

- LangGraph
- Qdrant 或 Chroma
- PostgreSQL
- Docker Compose
- DeepSeek、Qwen 或 OpenAI API

## 本地运行

当前推荐使用本机 Conda 环境：

```text
D:\Download\Coding\CondaData\envs_dirs\llm_env\python.exe
```

安装依赖：

```powershell
cd "D:\Code\codex\AI lab\rag-agent-system"
D:\Download\Coding\CondaData\envs_dirs\llm_env\python.exe -m pip install -r requirements.txt
```

启动服务：

```powershell
D:\Download\Coding\CondaData\envs_dirs\llm_env\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

打开：

```text
http://localhost:8000
http://localhost:8000/docs
http://localhost:8000/health
```

## 前端 Demo 联调说明

前端 Demo 页面位于 `personal-ai-homepage`：

```text
http://localhost:3000/demos/rag-agent
```

该页面会调用本项目的两个接口：

```text
GET  http://localhost:8000/health
POST http://localhost:8000/api/chat
```

联调前需要先启动本服务，再启动 `personal-ai-homepage`。

## 接口说明

### GET /

返回服务名称、运行状态和文档地址。

示例响应：

```json
{
  "name": "RAG Agent System MVP",
  "status": "running",
  "docs": "http://localhost:8000/docs",
  "health": "http://localhost:8000/health"
}
```

### GET /health

返回服务健康状态。

示例响应：

```json
{
  "status": "ok",
  "service": "rag-agent-system",
  "version": "0.1.0",
  "timestamp": "2026-05-19T00:00:00+00:00"
}
```

### POST /api/chat

发送一个问题，返回 mock 回答和 mock 引用来源。

请求示例：

```json
{
  "question": "这个系统现在支持什么？"
}
```

响应字段：

```text
answer   回答内容
sources  mock 引用来源
mode     当前模式，现阶段为 mock
```

PowerShell 测试命令：

```powershell
Invoke-RestMethod `
  -Uri "http://localhost:8000/api/chat" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"question":"这个系统现在支持什么？"}'
```

## 当前阶段目标

当前阶段不追求完整 RAG，而是先验证：

- 后端服务可以启动
- API 文档可以访问
- 个人主页 Demo 页面可以调用本服务
- 问答接口结构可以继续扩展
- 前端可以展示回答和引用来源

## 后续计划

- 增加文档上传接口
- 增加 PDF、Markdown、TXT 文本解析
- 增加文本切分模块
- 接入 Embedding 模型
- 接入 Qdrant 或 Chroma
- 实现真实 RAG 问答
- 返回引用来源
- 增加流式输出
- 接入 LangGraph Agent
- 使用 Docker Compose 部署

