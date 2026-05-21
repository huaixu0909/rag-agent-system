from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "data" / "chroma"
COLLECTION_NAME = "rag_chunks"


def chroma_available() -> bool:
    try:
        import chromadb  # noqa: F401
    except ImportError:
        return False
    return True


def get_collection():
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "Local RAG chunks for rag-agent-system",
            "hnsw:space": "cosine",
        },
    )


def normalize_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool | None]:
    normalized: dict[str, str | int | float | bool | None] = {}
    for key, value in metadata.items():
        if isinstance(value, list):
            normalized[key] = " / ".join(str(item) for item in value if item)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            normalized[key] = value
        else:
            normalized[key] = str(value)
    return normalized


def chunk_to_metadata(document: Any, chunk: Any) -> dict[str, str | int | float | bool | None]:
    return normalize_metadata(
        {
            "document_id": document.id,
            "document_filename": document.filename,
            "chunk_id": chunk.id,
            "chunk_index": chunk.index,
            "title": chunk.title,
            "section_path": chunk.section_path,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "strategy": chunk.strategy,
            "char_count": chunk.char_count,
            "token_estimate": chunk.token_estimate,
            "embedding_provider": getattr(chunk, "embedding_provider", "local_hash"),
        }
    )


def upsert_document_chunks(document: Any, chunks: list[Any]) -> int:
    if not chunks:
        return 0

    collection = get_collection()
    collection.upsert(
        ids=[chunk.id for chunk in chunks],
        documents=[chunk.content for chunk in chunks],
        embeddings=[chunk.embedding for chunk in chunks],
        metadatas=[chunk_to_metadata(document, chunk) for chunk in chunks],
    )
    return len(chunks)


def delete_document_chunks(document_id: str) -> int:
    collection = get_collection()
    existing = collection.get(where={"document_id": document_id})
    ids = existing.get("ids", [])
    if not ids:
        return 0

    collection.delete(ids=ids)
    return len(ids)


def query_chunks(question_embedding: list[float], top_k: int) -> list[dict[str, Any]]:
    collection = get_collection()
    result = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    ids = result.get("ids", [[]])[0]
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    items: list[dict[str, Any]] = []
    for index, chunk_id in enumerate(ids):
        distance = float(distances[index]) if index < len(distances) else 1.0
        score = max(0.0, min(1.0, 1.0 - distance))
        metadata = metadatas[index] or {}
        items.append(
            {
                "chunk_id": chunk_id,
                "content": documents[index] if index < len(documents) else "",
                "metadata": metadata,
                "score": round(score, 6),
            }
        )
    return items


def get_status() -> dict[str, Any]:
    collection = get_collection()
    return {
        "provider": "chroma",
        "available": True,
        "persist_path": str(CHROMA_DIR.relative_to(BASE_DIR)).replace("\\", "/"),
        "collection": COLLECTION_NAME,
        "chunk_count": collection.count(),
    }


def reset_collection() -> None:
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "Local RAG chunks for rag-agent-system",
            "hnsw:space": "cosine",
        },
    )
