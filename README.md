# rag-agent-system

企业知识库 RAG Agent 系统最小原型。

当前版本是 `v0.8`，已经支持文档上传、文本解析、结构增强 chunking、chunk embedding 保存、根据用户问题检索最相关 chunks，并通过 DeepSeek API 基于检索结果生成回答。

## 当前功能

- FastAPI 后端服务
- 文档上传接口：`POST /api/documents/upload`
- 文档列表接口：`GET /api/documents`
- 文档详情接口：`GET /api/documents/{document_id}`
- 文档删除接口：`DELETE /api/documents/{document_id}`
- 问题检索接口：`POST /api/search`
- LLM 问答接口：`POST /api/chat`
- Swagger API 文档：`GET /docs`
- 支持个人主页 `http://localhost:3000` 跨域访问

## v0.7 Chunking 流程

```text
1. 上传 .txt / .md / .pdf 文档
2. 后端解析为纯文本
3. 清洗空白、页码和 PDF 页标记
4. 按标题、章节、页码和段落生成语义单元
5. 优先按章节边界切分，章节过长时再做语义二次切分
6. 给相邻 chunk 保存 overlap_previous / overlap_next
7. 为 chunk + overlap 生成本地哈希 embedding
8. 用户提交问题后，计算 question embedding 和 chunk embedding 的余弦相似度
9. 返回相似度最高的 top_k 个 chunks
```

当前 embedding 是本地哈希向量，适合 MVP 验证流程。它不是生产级语义模型，后续可以替换为 `bge-small-zh`、Qwen Embedding、OpenAI Embedding，或写入 Qdrant / Chroma。

## v0.8 LLM 问答流程

```text
1. 用户提交问题和 top_k
2. 后端用 /api/search 同一套逻辑检索相关 chunks
3. 将问题、chunks、章节路径、页码和相似度拼入 RAG Prompt
4. 调用 DeepSeek Chat Completions
5. 返回 answer、sources 和 mode
6. 如果没有配置 API Key 或调用失败，自动降级为 retrieval_template
```

`.env` 配置：

```env
DEEPSEEK_API_KEY=你的 DeepSeek API Key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

`.env` 已加入 `.gitignore`，不要提交到 GitHub。

## Chunk 字段

```text
id                    chunk id
document_id           所属文档 id
index                 chunk 序号
content               chunk 正文
char_count            字符数
token_estimate        token 粗略估算
title                 最近标题
heading_level         标题层级
section_path          章节路径
page_start            起始页码
page_end              结束页码
strategy              section_semantic / section_semantic_split / length_fallback
semantic_break_score  与前一语义单元的相似度
overlap_previous      上一个 chunk 的尾部上下文
overlap_next          下一个 chunk 的头部上下文
embedding             本地哈希向量
```

## API 示例

### 上传文档

```powershell
Invoke-RestMethod `
  -Uri "http://localhost:8000/api/documents/upload" `
  -Method Post `
  -Form @{ file = Get-Item "D:\path\to\demo.pdf" }
```

### 查询文档列表

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/api/documents" -Method Get
```

### 删除文档

```powershell
Invoke-RestMethod `
  -Uri "http://localhost:8000/api/documents/doc_xxxxxxxxxxxx" `
  -Method Delete
```

删除会同时移除文档元数据、原始上传文件、解析文本和 chunks 文件。

### 根据问题检索 chunks

```powershell
Invoke-RestMethod `
  -Uri "http://localhost:8000/api/search" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"question":"这个文档里和项目经验相关的内容是什么？","top_k":5}'
```

返回核心字段：

```text
total_chunks              本次扫描的 chunk 总数
results[].score           问题和 chunk 的相似度
results[].content         命中的 chunk 内容
results[].strategy        chunk 切分策略
results[].section_path    chunk 所属章节路径
results[].page_start      chunk 起始页
results[].page_end        chunk 结束页
results[].token_estimate  token 估算
```

### 生成模板回答

```powershell
Invoke-RestMethod `
  -Uri "http://localhost:8000/api/chat" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"question":"这个文档主要讲了什么？","top_k":5}'
```

如果 DeepSeek 调用成功，返回 `mode=deepseek`。如果没有 API Key 或网络/API 调用失败，返回 `mode=retrieval_template`。

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
data/documents.json 文档元数据
```

`data/` 已加入 `.gitignore`，本地测试文档不会被提交到 GitHub。

## 前端 Demo

```text
http://localhost:3000/demos/rag-agent
```

前端会展示：

```text
文档上传
文档删除
文档列表
解析文本预览
结构增强 chunks
章节路径
页码信息
overlap 上下文
问题检索结果
问题和 chunk 的相似度分数
模板回答和 sources
```

## v0.7 验收标准

- 上传 `.txt`、`.md`、`.pdf` 后能生成 chunks
- chunk JSON 中包含 `embedding`
- chunk JSON 中包含 `section_path`、`token_estimate`、`page_start`、`page_end`
- chunk JSON 中包含 `overlap_previous` 和 `overlap_next`
- PDF 文档的页码信息可以进入 chunk metadata
- 过长章节会被二次切分为 `section_semantic_split`
- `POST /api/search` 可以返回 top_k 个相似 chunks
- 检索结果包含文档名、chunk 编号、相似度、章节路径、页码、chunk 内容
- 个人主页 RAG Demo 可以展示检索结果和 chunk metadata
- `/api/chat` 能优先调用 DeepSeek 基于检索结果生成回答
- DeepSeek 不可用时 `/api/chat` 能自动降级为模板回答

## 后续计划

- 替换为真实 embedding 模型
- 接入 Qdrant 或 Chroma
- 为检索增加文档过滤、分数阈值和 rerank
- 支持流式输出
- 支持多模型切换
