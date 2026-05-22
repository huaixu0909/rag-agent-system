# rag-agent-system

企业知识库 RAG Agent 系统最小原型。

当前版本是 `v1.9`，已经支持文档去重、文档摘要、手动标签、知识库概览问答、多轮对话、混合检索 rerank、异步入库任务队列、SQLite 文档元数据存储、文档上传、批量上传、上传解析进度反馈、后端分页文档库、文本解析、结构增强 chunking、JSON chunks 留存、Chroma 本地向量库检索、Qwen Embedding、相似度阈值拒答，以及 LangGraph + LangChain + DeepSeek RAG 问答。

## 当前功能

- FastAPI 后端服务
- SQLite 文档元数据存储：`data/rag_agent.db`
- 单文件上传接口：`POST /api/documents/upload`
- 异步批量上传接口：`POST /api/documents/upload/batch`
- 入库任务进度接口：`GET /api/ingest-tasks/{task_id}`
- 文档分页列表接口：`GET /api/documents?page=1&page_size=10`
- 文档详情接口：`GET /api/documents/{document_id}`
- 文档标签接口：`PATCH /api/documents/{document_id}/tags`
- 文档删除接口：`DELETE /api/documents/{document_id}`
- 混合检索接口：`POST /api/search`
- 多轮对话接口：`POST /api/chat`
- 知识库概览问答：通过 `POST /api/chat` 自动识别“当前知识库有哪些内容”等问题
- 会话消息接口：`GET /api/chat/sessions/{session_id}/messages`
- 向量库状态接口：`GET /api/vector-store/status`
- 向量库重建接口：`POST /api/vector-store/rebuild`
- Swagger API 文档：`GET /docs`

## v1.9 知识库管理增强

文档入库现在会写入以下元数据：

```text
content_hash   文件 SHA-256，用于内容重复检测
summary        入库时自动生成的文档摘要
tags           手动维护的文档标签 JSON
```

重复上传规则：

```text
同名文件       标记为 duplicate
内容 hash 相同 标记为 duplicate
```

批量上传接口会在任务文件项中返回 `duplicate` 状态，并通过 `error` 字段说明重复原因。重复文件不会继续解析、切分、embedding 或写入 Chroma。

手动更新标签：

```http
PATCH /api/documents/{document_id}/tags
Content-Type: application/json

{
  "tags": ["简历", "项目文档", "面试资料"]
}
```

删除文档时会同步清理：

```text
SQLite documents 记录
Chroma 向量索引
data/uploads 原始文件
data/parsed 解析文本
data/chunks JSON chunks
```

知识库概览问答会优先使用每份文档的 `summary`，因此“当前知识库有哪些内容”会比直接读取首个 chunk 更稳定。

## v1.8 知识库概览问答

`/api/chat` 现在会先识别知识库概览类问题，例如：

```text
当前知识库有哪些内容？
我上传了哪些文档？
文档库里有什么资料？
当前资料库收录了哪些文件？
```

命中这类意图后，系统不会走普通 chunk 相似度检索，而是直接读取 SQLite 中的文档元数据，并结合每个文档的少量 chunk / 解析文本生成稳定的知识库概览。

返回结果会包含：

```json
{
  "mode": "knowledge_overview",
  "graph_path": ["knowledge_overview"],
  "overview": {
    "document_count": 3,
    "total_chunks": 42,
    "total_char_count": 18000,
    "documents": []
  }
}
```

这样可以避免“当前知识库有哪些内容”被当成普通语义检索问题，从而命中无关 chunk。

## v1.7 多轮对话

`/api/chat` 现在支持 `session_id`：

```json
{
  "question": "第二点展开讲讲",
  "top_k": 5,
  "score_threshold": 0.2,
  "session_id": "session_xxxxxxxxxxxx"
}
```

如果没有传 `session_id`，后端会自动创建新会话，并在响应中返回：

```json
{
  "session_id": "session_xxxxxxxxxxxx",
  "rewritten_question": "包含最近会话上下文的检索问题",
  "messages": []
}
```

SQLite 新增表：

```text
chat_sessions   会话主表
chat_messages   user / assistant 消息记录
```

检索前会把最近几轮对话拼入 `rewritten_question`，让“这个、它、第二点、继续展开”等追问可以带上上下文再进入 RAG 检索。

查询会话消息：

```text
GET /api/chat/sessions/{session_id}/messages
```

## v1.6 检索质量优化

`/api/search` 已从单一向量分数升级为混合 rerank：

```text
1. 生成问题 embedding
2. Chroma 优先召回更多候选 chunks
3. 提取查询关键词、中文 bigram / trigram、英文技术词
4. 对候选 chunk 计算关键词覆盖、短语命中、标题命中、章节命中
5. 合并 vector_score + lexical_score + structural_boost
6. 按 rerank_score 排序，再应用 score_threshold
```

返回结果新增：

```json
{
  "retrieval_strategy": "hybrid_rerank",
  "query_terms": ["rag", "知识", "知识库"],
  "results": [
    {
      "score": 0.72,
      "vector_score": 0.66,
      "lexical_score": 0.84,
      "rerank_score": 0.72
    }
  ]
}
```

分数含义：

```text
vector_score   向量相似度
lexical_score  关键词、短语、标题、章节命中分
rerank_score   最终排序分，也是 score_threshold 使用的分数
```

这一步不需要新增 API Key。后续如果要进一步提升效果，可以接入真实 reranker 模型，例如 bge-reranker、Qwen Reranker 或 Cohere Rerank。

## v1.5 异步入库任务队列

批量上传已经从“请求内同步解析”改为“任务化后台入库”：

```text
POST /api/documents/upload/batch
-> 保存上传文件
-> 创建 task_id
-> 立即返回任务状态
-> 后台执行 parsing / chunking / embedding / indexing
```

查询任务进度：

```text
GET /api/ingest-tasks/{task_id}
```

任务状态：

```text
queued          已创建，等待后台执行
running         正在执行
completed       全部成功
partial_failed  部分成功、部分失败
failed          全部失败
```

单文件阶段：

```text
uploaded -> parsing -> chunking -> embedding -> indexing -> indexed
failed
```

SQLite 新增表：

```text
ingest_tasks       入库任务主表
ingest_task_files  单文件任务明细
```

前端 RAG Demo 会轮询任务接口，并展示每个文件的当前阶段。这样大文件解析、embedding 和写入 Chroma 不再阻塞上传接口响应。

## v1.4 SQLite 文档元数据

文档元数据已经从 `data/documents.json` 迁移到 SQLite：

```text
data/rag_agent.db
```

SQLite 当前负责保存：

```text
文档 id
原始文件名
文件类型
原始文件路径
解析文本路径
chunks 路径
字符数
chunk 数
创建时间
入库任务状态
```

系统启动或首次访问接口时，会自动创建数据库表。如果本地还存在旧的 `data/documents.json`，系统会自动导入一次，并写入迁移标记，避免后续删除文档后又被旧 JSON 重新导入。

当前存储分工：

```text
SQLite             文档元数据、分页列表、详情索引
data/uploads       原始上传文件
data/parsed        解析后的纯文本
data/chunks        可调试 JSON chunks
Chroma             向量检索索引
```

## v1.3 LangGraph RAG Chat

`/api/chat` 现在优先走 LangGraph 工作流：

```text
START
-> retrieve            检索 Chroma / JSON fallback
-> 条件分支
   -> reject_answer    没有足够相关资料，直接拒答
   -> generate_answer  有相关资料，调用 LangChain + DeepSeek 生成答案
-> END
```

返回结果会包含：

```json
{
  "workflow": "langgraph",
  "graph_path": ["retrieve", "generate_answer"],
  "mode": "langgraph_deepseek"
}
```

如果本地还没有安装 LangGraph，接口会自动回退到原来的手动 RAG 流程，并返回：

```json
{
  "workflow": "manual"
}
```

安装或更新依赖：

```powershell
D:\Download\Coding\CondaData\envs_dirs\llm_env\python.exe -m pip install -r requirements.txt
```

## v1.2 知识库管理体验优化

文档库列表已经改为后端分页，适合后续上传大量资料：

```text
GET /api/documents?page=1&page_size=10
```

返回结构：

```json
{
  "items": [],
  "total": 25,
  "page": 1,
  "page_size": 10,
  "total_pages": 3,
  "total_chunks": 120
}
```

批量上传接口：

```text
POST /api/documents/upload/batch
```

每个文件都会返回独立状态，前端可以展示上传、解析、入库或失败结果。当前 v1.2 是请求级进度反馈；如果后续要支持超大文件和后台任务，可以升级为任务队列 + 轮询进度。

## v1.0 Chroma 向量数据库

当前检索层已经从“遍历 JSON chunks”升级为：

```text
上传文档
-> 解析文本
-> 结构增强 chunking
-> 保存 JSON chunks
-> 写入 Chroma collection
-> /api/search 优先使用 Chroma similarity search
-> Chroma 不可用时回退到 JSON 遍历检索
```

保留 JSON chunks 的原因：

```text
JSON chunks：可读、可调试、适合文档详情页展示
Chroma：负责向量相似度检索
```

本地向量库目录：

```text
data/chroma
```

`data/` 已加入 `.gitignore`，Chroma 本地索引不会提交到 GitHub。

## v0.9 LangChain RAG Chat

`/api/chat` 当前执行顺序：

```text
1. 根据用户问题检索 top_k chunks
2. 将 chunks、章节路径、页码、相似度整理为 context
3. 使用 LangChain ChatPromptTemplate 构建 RAG Prompt
4. 使用 langchain-openai 的 ChatOpenAI 调用 DeepSeek OpenAI-compatible API
5. 使用 StrOutputParser 输出文本回答
6. 如果 LangChain 或依赖不可用，降级为旧版 DeepSeek 直连
7. 如果 DeepSeek 也不可用，降级为 retrieval_template
```

返回的 `mode`：

```text
langchain_deepseek   LangChain + DeepSeek 调用成功
deepseek             旧版 DeepSeek 直连兜底成功
retrieval_template   LLM 不可用，返回检索模板回答
```

## Chunking 流程

```text
1. 上传 .txt / .md / .pdf 文档
2. 后端解析为纯文本
3. 清洗空白、页码和 PDF 页标记
4. 按标题、章节、页码和段落生成语义单元
5. 优先按章节边界切分，章节过长时再做语义二次切分
6. 给相邻 chunk 保存 overlap_previous / overlap_next
7. 为 chunk + overlap 生成本地哈希 embedding
8. 写入 JSON chunks 和 Chroma
```

当前 embedding 是本地哈希向量，适合 MVP 验证流程。它不是生产级语义模型，后续可以替换为 `bge-small-zh`、Qwen Embedding、OpenAI Embedding。

## v1.1 Qwen Embedding 与严格检索

系统现在支持 Qwen Embedding。配置 API Key 后，上传和重建索引会优先使用 Qwen 生成 embedding；未配置时自动回退本地哈希 embedding。

`.env` 配置示例：

```env
DASHSCOPE_API_KEY=你的 DashScope API Key
QWEN_EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_EMBEDDING_MODEL=text-embedding-v4
QWEN_EMBEDDING_DIMENSIONS=1024
```

也可以使用：

```env
QWEN_EMBEDDING_API_KEY=你的 DashScope API Key
```

配置 Qwen Embedding 后，请执行一次：

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/api/vector-store/rebuild" -Method Post
```

这样已有 JSON chunks 会重新生成 Qwen embedding 并写入 Chroma。

检索和问答支持相似度阈值：

```json
{
  "question": "这个文档里和项目经验相关的内容是什么？",
  "top_k": 5,
  "score_threshold": 0.2
}
```

如果没有任何 chunk 达到阈值，`/api/chat` 会直接返回：

```text
当前知识库中没有足够信息回答这个问题。
```

## 环境变量

`.env` 配置：

```env
DEEPSEEK_API_KEY=你的 DeepSeek API Key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
DASHSCOPE_API_KEY=你的 DashScope API Key
QWEN_EMBEDDING_MODEL=text-embedding-v4
```

`.env` 已加入 `.gitignore`，不要提交到 GitHub。

## API 示例

### 查询向量库状态

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/api/vector-store/status" -Method Get
```

返回示例：

```json
{
  "provider": "chroma",
  "available": true,
  "persist_path": "data/chroma",
  "collection": "rag_chunks",
  "chunk_count": 12,
  "embedding_provider": "qwen"
}
```

### 重建向量库

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/api/vector-store/rebuild" -Method Post
```

用于把已有 `data/chunks/*.json` 重新写入 Chroma。

### 上传文档

```powershell
Invoke-RestMethod `
  -Uri "http://localhost:8000/api/documents/upload" `
  -Method Post `
  -Form @{ file = Get-Item "D:\path\to\demo.pdf" }
```

### 批量上传文档

```powershell
Invoke-RestMethod `
  -Uri "http://localhost:8000/api/documents/upload/batch" `
  -Method Post `
  -Form @{
    files = @(
      Get-Item "D:\path\to\a.pdf"
      Get-Item "D:\path\to\b.md"
    )
  }
```

### 分页查看文档库

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/api/documents?page=1&page_size=10" -Method Get
```

### 根据问题检索 chunks

```powershell
Invoke-RestMethod `
  -Uri "http://localhost:8000/api/search" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"question":"这个文档里和项目经验相关的内容是什么？","top_k":5,"score_threshold":0.2}'
```

如果 Chroma 命中，返回：

```text
mode = chroma
```

如果 Chroma 不可用，会回退：

```text
mode = local_hash_embedding
```

### LangChain RAG 问答

```powershell
Invoke-RestMethod `
  -Uri "http://localhost:8000/api/chat" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"question":"这个文档主要讲了什么？","top_k":5,"score_threshold":0.2}'
```

## 本地运行

```powershell
cd "D:\Code\codex\AI lab\rag-agent-system"
D:\Download\Coding\CondaData\envs_dirs\llm_env\python.exe -m pip install -r requirements.txt
D:\Download\Coding\CondaData\envs_dirs\llm_env\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

打开：

```text
http://localhost:8000
http://localhost:8000/docs
```

## 数据目录

```text
data/uploads        原始上传文件
data/parsed         解析后的纯文本
data/chunks         文本切分后的 chunks、metadata、overlap 和 embedding
data/chroma         Chroma 本地向量库
data/rag_agent.db   SQLite 文档元数据
data/documents.json v1.4 之前的旧元数据文件，仅用于一次性迁移
```

## 前端 Demo

```text
http://localhost:3000/demos/rag-agent
```

前端会展示：

```text
文档上传
文档删除
文档列表
Chroma / JSON fallback 状态
解析文本预览
结构增强 chunks
问题检索结果
回答和 sources
```

## v1.0 验收标准

- 上传新文档后，chunks 同时保存到 JSON 和 Chroma
- `GET /api/vector-store/status` 返回 Chroma 状态
- `POST /api/vector-store/rebuild` 能把已有 JSON chunks 重建进 Chroma
- `POST /api/search` 优先返回 `mode=chroma`
- 删除文档时同步删除 Chroma 中对应 chunks
- Chroma 不可用时保留 JSON fallback
- LangChain / DeepSeek 问答链路保持兼容

## v1.1 验收标准

- 配置 `DASHSCOPE_API_KEY` 后，系统优先使用 Qwen Embedding
- `GET /api/vector-store/status` 返回 `embedding_provider=qwen`
- `POST /api/vector-store/rebuild` 能用 Qwen embedding 重建 Chroma
- `/api/search` 支持 `score_threshold`
- `/api/chat` 在无足够相关资料时返回“当前知识库中没有足够信息回答这个问题。”
- Prompt 明确要求只基于资料回答，并引用来源编号

## 后续计划

- 替换为真实 embedding 模型
- 为检索增加文档过滤、分数阈值和 rerank
- 支持流式输出
- 使用 LangGraph 编排多步骤 Agent Workflow
