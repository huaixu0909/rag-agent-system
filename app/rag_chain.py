import os
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


def format_context_chunks(chunks: list[dict[str, Any]]) -> str:
    context_blocks: list[str] = []

    for index, chunk in enumerate(chunks, start=1):
        section_path = chunk.get("section_path") or []
        section_text = " / ".join(section_path) if section_path else "未识别章节"

        page_start = chunk.get("page_start")
        page_end = chunk.get("page_end")
        page_text = "无页码"
        if page_start:
            page_text = f"第 {page_start} 页"
            if page_end and page_end != page_start:
                page_text = f"第 {page_start}-{page_end} 页"

        context_blocks.append(
            "\n".join(
                [
                    f"[来源 {index}]",
                    f"文档：{chunk.get('document_filename', '')}",
                    f"Chunk：{chunk.get('chunk_index', '')}",
                    f"相似度：{float(chunk.get('score') or 0):.3f}",
                    f"章节：{section_text}",
                    f"页码：{page_text}",
                    "内容：",
                    str(chunk.get("content") or ""),
                ]
            )
        )

    return "\n\n".join(context_blocks)


def generate_rag_answer_with_langchain(question: str, chunks: list[dict[str, Any]]) -> str | None:
    load_env_file()
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        return None

    try:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI
    except ImportError:
        return None

    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    try:
        llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=0.2,
            max_tokens=1200,
        )
    except TypeError:
        llm = ChatOpenAI(
            model=model,
            openai_api_key=api_key,
            openai_api_base=base_url,
            temperature=0.2,
            max_tokens=1200,
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一个严谨的中文 RAG 问答助手。你只能根据参考资料回答问题。"
                "如果资料不足，请明确说明“当前文档中没有足够信息”。不要编造。",
            ),
            (
                "human",
                "用户问题：\n{question}\n\n"
                "参考资料：\n{context}\n\n"
                "请先给出直接结论，再给出依据。最后用简短列表列出引用来源编号。",
            ),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    try:
        answer = chain.invoke(
            {
                "question": question,
                "context": format_context_chunks(chunks),
            }
        )
    except Exception:
        return None

    return str(answer).strip() or None
