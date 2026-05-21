import json
import math
import os
import re
import urllib.error
import urllib.request
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
EMBEDDING_DIM = 256


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


def stable_hash_index(token: str) -> int:
    import hashlib

    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % EMBEDDING_DIM


def text_tokens(text: str) -> list[str]:
    lowered = text.lower()
    words = re.findall(r"[a-z0-9]+", lowered)
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    chinese_bigrams = [f"{a}{b}" for a, b in zip(chinese_chars, chinese_chars[1:])]
    return words + chinese_chars + chinese_bigrams


def embed_text_local(text: str) -> list[float]:
    vector = [0.0] * EMBEDDING_DIM
    for token in text_tokens(text):
        vector[stable_hash_index(token)] += 1.0

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector

    return [value / norm for value in vector]


def qwen_embedding_config() -> tuple[str, str, str]:
    load_env_file()
    api_key = (
        os.getenv("QWEN_EMBEDDING_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("QWEN_API_KEY")
        or ""
    ).strip()
    base_url = os.getenv(
        "QWEN_EMBEDDING_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ).rstrip("/")
    model = os.getenv("QWEN_EMBEDDING_MODEL", "text-embedding-v4")
    return api_key, base_url, model


def qwen_embedding_enabled() -> bool:
    api_key, _, _ = qwen_embedding_config()
    return bool(api_key)


def embed_text_qwen(text: str) -> list[float] | None:
    api_key, base_url, model = qwen_embedding_config()
    if not api_key:
        return None

    payload = {
        "model": model,
        "input": text,
        "encoding_format": "float",
    }
    dimensions = os.getenv("QWEN_EMBEDDING_DIMENSIONS", "").strip()
    if dimensions:
        payload["dimensions"] = int(dimensions)
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url=f"{base_url}/embeddings",
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

    items = data.get("data") or []
    if not items:
        return None

    embedding = items[0].get("embedding")
    if not isinstance(embedding, list):
        return None

    return [float(value) for value in embedding]


def embed_text(text: str) -> list[float]:
    qwen_embedding = embed_text_qwen(text)
    if qwen_embedding:
        return qwen_embedding
    return embed_text_local(text)


def embedding_provider() -> str:
    return "qwen" if qwen_embedding_enabled() else "local_hash"
