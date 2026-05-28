import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent


def load_env_file() -> None:
    env_path = BASE_DIR / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def build_retrieval_template_answer(question: str, results: list[Any]) -> str:
    summary_lines = [
        f"{index}. 《{item.document_filename}》chunk {item.chunk_index}，相似度 {item.score:.3f}"
        for index, item in enumerate(results, start=1)
    ]
    return (
        "已完成知识库检索，但没有成功调用 LLM。以下是可作为回答依据的命中文档片段：\n"
        + "\n".join(summary_lines)
        + f"\n\n用户问题：{question}"
    )


def build_rag_prompt(question: str, results: list[Any]) -> str:
    context_blocks = []
    for index, item in enumerate(results, start=1):
        page_text = "无页码"
        if item.page_start:
            page_text = f"第 {item.page_start} 页"
            if item.page_end and item.page_end != item.page_start:
                page_text = f"第 {item.page_start}-{item.page_end} 页"

        section_text = " / ".join(item.section_path) if item.section_path else "未识别章节"
        context_blocks.append(
            "\n".join(
                [
                    f"[来源 {index}]",
                    f"文档：{item.document_filename}",
                    f"chunk：{item.chunk_index}",
                    f"相似度：{item.score:.3f}",
                    f"章节：{section_text}",
                    f"页码：{page_text}",
                    "内容：",
                    item.content,
                ]
            )
        )

    return (
        "你是一个严谨的中文知识库问答助手。请只根据给定资料回答用户问题。\n"
        "如果资料不足，请明确回答：当前知识库中没有足够信息回答这个问题。\n"
        "禁止使用资料之外的知识补全答案，禁止编造。\n"
        "回答必须先给结论，再给依据，并在依据中引用来源编号。\n\n"
        f"用户问题：\n{question}\n\n"
        "参考资料：\n"
        + "\n\n".join(context_blocks)
    )


def call_deepseek_chat(question: str, results: list[Any]) -> str | None:
    load_env_file()
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        return None

    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    url = f"{base_url}/chat/completions"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "你是可靠的 RAG 问答助手，只能基于参考资料回答，并必须引用来源编号。",
            },
            {
                "role": "user",
                "content": build_rag_prompt(question, results),
            },
        ],
        "temperature": 0.2,
        "max_tokens": 1200,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url=url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None

    choices = data.get("choices") or []
    if not choices:
        return None

    message = choices[0].get("message") or {}
    answer = str(message.get("content") or "").strip()
    return answer or None
