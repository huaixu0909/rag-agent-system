import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pypdf import PdfReader

from app.embeddings import embed_text, embedding_provider
from app.rag_graph import NO_ENOUGH_CONTEXT_ANSWER, run_langgraph_rag_chat
from app.rag_chain import generate_rag_answer_with_langchain
from app.security import rate_limit, require_admin
from app.vector_store import (
    chroma_available,
    delete_document_chunks,
    get_status as get_vector_store_status,
    query_chunks,
    reset_collection,
    upsert_document_chunks,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
PARSED_DIR = DATA_DIR / "parsed"
CHUNKS_DIR = DATA_DIR / "chunks"
CHROMA_DIR = DATA_DIR / "chroma"
DOCUMENTS_FILE = DATA_DIR / "documents.json"
DATABASE_FILE = DATA_DIR / "rag_agent.db"
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}

APP_VERSION = "1.9.0"
EMBEDDING_DIM = 256
TARGET_CHUNK_SIZE = 1200
MAX_CHUNK_SIZE = 1800
MIN_CHUNK_SIZE = 300
MIN_HEADING_CHUNK_SIZE = 80
CHUNK_OVERLAP_CHARS = 180
DETAIL_PREVIEW_CHARS = 1200
DETAIL_MAX_CHUNKS = 20
SEARCH_TOP_K_DEFAULT = 5
SEARCH_TOP_K_MAX = 20
SEARCH_SCORE_THRESHOLD_DEFAULT = 0.2
RERANK_CANDIDATE_MULTIPLIER = 6
RERANK_CANDIDATE_MAX = 80
OVERVIEW_PREVIEW_CHARS = 260
OVERVIEW_MAX_DOCUMENTS_IN_ANSWER = 20
DOCUMENT_SUMMARY_CHARS = 360
DOCUMENT_TAG_MAX_COUNT = 12
DOCUMENT_TAG_MAX_LENGTH = 24

ChunkStrategy = Literal[
    "semantic",
    "semantic_split",
    "length_fallback",
    "structure",
    "structure_split",
    "section_semantic",
    "section_semantic_split",
]

app = FastAPI(
    title="RAG Agent System MVP",
    description="A local prototype for document upload, parsing, semantic chunking, and question-based retrieval.",
    version=APP_VERSION,
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


class DocumentRecord(BaseModel):
    id: str
    filename: str
    file_type: str
    stored_path: str
    parsed_path: str = ""
    chunks_path: str = ""
    content_hash: str = ""
    summary: str = ""
    tags: list[str] = Field(default_factory=list)
    char_count: int
    chunk_count: int = 0
    created_at: str


class DocumentListResponse(BaseModel):
    items: list[DocumentRecord]
    total: int
    page: int
    page_size: int
    total_pages: int
    total_chunks: int


class UploadProgressItem(BaseModel):
    filename: str
    status: Literal["indexed", "failed"]
    stage: Literal["uploaded", "parsing", "chunking", "embedding", "indexed", "failed"]
    document: DocumentRecord | None = None
    error: str = ""


class BatchUploadResponse(BaseModel):
    total: int
    succeeded: int
    failed: int
    items: list[UploadProgressItem]


IngestTaskStatus = Literal["queued", "running", "completed", "failed", "partial_failed"]
IngestFileStatus = Literal["queued", "running", "indexed", "failed", "duplicate"]
IngestFileStage = Literal[
    "uploaded",
    "queued",
    "parsing",
    "chunking",
    "embedding",
    "indexing",
    "indexed",
    "duplicate",
    "failed",
]


class IngestTaskFileResponse(BaseModel):
    file_id: str
    filename: str
    status: IngestFileStatus
    stage: IngestFileStage
    document_id: str = ""
    document: DocumentRecord | None = None
    duplicate_document: DocumentRecord | None = None
    char_count: int = 0
    chunk_count: int = 0
    error: str = ""


class IngestTaskResponse(BaseModel):
    task_id: str
    status: IngestTaskStatus
    total: int
    succeeded: int
    failed: int
    created_at: str
    updated_at: str
    completed_at: str = ""
    items: list[IngestTaskFileResponse]


class DocumentChunk(BaseModel):
    id: str
    document_id: str
    index: int
    content: str
    char_count: int
    title: str = ""
    heading_level: int | None = None
    strategy: ChunkStrategy = "length_fallback"
    semantic_break_score: float | None = None
    section_path: list[str] = Field(default_factory=list)
    token_estimate: int = 0
    page_start: int | None = None
    page_end: int | None = None
    overlap_previous: str = ""
    overlap_next: str = ""
    embedding: list[float] = Field(default_factory=list)
    embedding_provider: str = "local_hash"


class DocumentDetail(BaseModel):
    document: DocumentRecord
    text_preview: str
    preview_char_count: int
    chunks: list[DocumentChunk]
    returned_chunk_count: int


class DeleteDocumentResponse(BaseModel):
    deleted: bool
    document_id: str
    filename: str
    removed_files: list[str]
    vector_chunks_deleted: int = 0
    sqlite_deleted: bool = True


class UpdateDocumentTagsRequest(BaseModel):
    tags: list[str] = Field(default_factory=list, max_length=DOCUMENT_TAG_MAX_COUNT)


class SearchRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=SEARCH_TOP_K_DEFAULT, ge=1, le=SEARCH_TOP_K_MAX)
    score_threshold: float = Field(default=SEARCH_SCORE_THRESHOLD_DEFAULT, ge=0.0, le=1.0)


class SearchResult(BaseModel):
    document_id: str
    document_filename: str
    chunk_id: str
    chunk_index: int
    title: str = ""
    score: float
    vector_score: float = 0.0
    lexical_score: float = 0.0
    rerank_score: float = 0.0
    content: str
    char_count: int
    strategy: ChunkStrategy
    section_path: list[str] = Field(default_factory=list)
    token_estimate: int = 0
    page_start: int | None = None
    page_end: int | None = None


class SearchResponse(BaseModel):
    question: str
    top_k: int
    score_threshold: float
    total_chunks: int
    results: list[SearchResult]
    mode: Literal["chroma", "local_hash_embedding"]
    retrieval_strategy: Literal["hybrid_rerank"] = "hybrid_rerank"
    query_terms: list[str] = Field(default_factory=list)


class VectorStoreStatusResponse(BaseModel):
    provider: Literal["chroma"]
    available: bool
    persist_path: str
    collection: str
    chunk_count: int
    embedding_provider: str = "local_hash"


class VectorStoreRebuildResponse(BaseModel):
    rebuilt: bool
    provider: Literal["chroma"]
    document_count: int
    chunk_count: int


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=SEARCH_TOP_K_DEFAULT, ge=1, le=SEARCH_TOP_K_MAX)
    score_threshold: float = Field(default=SEARCH_SCORE_THRESHOLD_DEFAULT, ge=0.0, le=1.0)
    session_id: str | None = Field(default=None, max_length=80)


class Source(BaseModel):
    title: str
    content: str
    document_id: str = ""
    chunk_id: str = ""
    score: float | None = None
    page_start: int | None = None
    page_end: int | None = None
    section_path: list[str] = Field(default_factory=list)


class KnowledgeOverviewDocument(BaseModel):
    document_id: str
    filename: str
    file_type: str
    char_count: int
    chunk_count: int
    created_at: str
    preview: str = ""
    summary: str = ""
    tags: list[str] = Field(default_factory=list)


class KnowledgeOverview(BaseModel):
    document_count: int
    total_chunks: int
    total_char_count: int
    documents: list[KnowledgeOverviewDocument]
    truncated: bool = False


class ChatMessage(BaseModel):
    id: str
    session_id: str
    role: Literal["user", "assistant"]
    content: str
    created_at: str


class ChatResponse(BaseModel):
    session_id: str = ""
    rewritten_question: str = ""
    answer: str
    sources: list[Source]
    mode: Literal[
        "langgraph_deepseek",
        "langchain_deepseek",
        "deepseek",
        "retrieval_template",
        "knowledge_overview",
    ]
    retrieval_mode: Literal["chroma", "local_hash_embedding"] = "local_hash_embedding"
    score_threshold: float = SEARCH_SCORE_THRESHOLD_DEFAULT
    workflow: Literal["langgraph", "manual"] = "manual"
    graph_path: list[str] = Field(default_factory=list)
    messages: list[ChatMessage] = Field(default_factory=list)
    overview: KnowledgeOverview | None = None


@dataclass
class Heading:
    title: str
    level: int


@dataclass
class SemanticUnit:
    content: str
    title: str = ""
    heading_level: int | None = None
    is_heading: bool = False
    section_path: list[str] | None = None
    page_start: int | None = None
    page_end: int | None = None


def ensure_data_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    initialize_database()


def open_database() -> sqlite3.Connection:
    connection = sqlite3.connect(DATABASE_FILE)
    connection.row_factory = sqlite3.Row
    return connection


def ensure_table_columns(
    connection: sqlite3.Connection,
    table_name: str,
    columns: dict[str, str],
) -> None:
    existing_columns = {
        str(row["name"])
        for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    }

    for column_name, column_definition in columns.items():
        if column_name not in existing_columns:
            connection.execute(
                f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}"
            )


def ensure_document_columns(connection: sqlite3.Connection) -> None:
    ensure_table_columns(
        connection,
        "documents",
        {
            "content_hash": "TEXT NOT NULL DEFAULT ''",
            "summary": "TEXT NOT NULL DEFAULT ''",
            "tags": "TEXT NOT NULL DEFAULT '[]'",
        },
    )


def ensure_ingest_task_file_columns(connection: sqlite3.Connection) -> None:
    ensure_table_columns(
        connection,
        "ingest_task_files",
        {
            "duplicate_document_id": "TEXT NOT NULL DEFAULT ''",
        },
    )


def initialize_database() -> None:
    with open_database() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                file_type TEXT NOT NULL,
                stored_path TEXT NOT NULL,
                parsed_path TEXT NOT NULL DEFAULT '',
                chunks_path TEXT NOT NULL DEFAULT '',
                content_hash TEXT NOT NULL DEFAULT '',
                summary TEXT NOT NULL DEFAULT '',
                tags TEXT NOT NULL DEFAULT '[]',
                char_count INTEGER NOT NULL DEFAULT 0,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
            """
        )
        ensure_document_columns(connection)
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at DESC)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_filename ON documents(filename)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(content_hash)"
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS app_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS ingest_tasks (
                task_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                total INTEGER NOT NULL DEFAULT 0,
                succeeded INTEGER NOT NULL DEFAULT 0,
                failed INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed_at TEXT NOT NULL DEFAULT ''
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS ingest_task_files (
                file_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                document_id TEXT NOT NULL DEFAULT '',
                filename TEXT NOT NULL,
                file_type TEXT NOT NULL DEFAULT '',
                stored_path TEXT NOT NULL DEFAULT '',
                duplicate_document_id TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL,
                stage TEXT NOT NULL,
                char_count INTEGER NOT NULL DEFAULT 0,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                error TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(task_id) REFERENCES ingest_tasks(task_id)
            )
            """
        )
        ensure_ingest_task_file_columns(connection)
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_ingest_task_files_task_id ON ingest_task_files(task_id)"
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES chat_sessions(session_id)
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON chat_messages(session_id, created_at ASC)"
        )
        connection.commit()

    migrate_documents_json_to_sqlite()


def migrate_documents_json_to_sqlite() -> None:
    with open_database() as connection:
        migrated = connection.execute(
            "SELECT value FROM app_meta WHERE key = ?",
            ("documents_json_migrated",),
        ).fetchone()
        if migrated is not None:
            return

    if not DOCUMENTS_FILE.exists():
        mark_documents_json_migrated()
        return

    try:
        raw_documents = json.loads(DOCUMENTS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        mark_documents_json_migrated()
        return

    if not isinstance(raw_documents, list):
        mark_documents_json_migrated()
        return

    documents: list[DocumentRecord] = []
    for item in raw_documents:
        try:
            documents.append(DocumentRecord(**item))
        except Exception:
            continue

    if not documents:
        mark_documents_json_migrated()
        return

    with open_database() as connection:
        for document in documents:
            connection.execute(
                """
                INSERT OR IGNORE INTO documents (
                    id, filename, file_type, stored_path, parsed_path,
                    chunks_path, content_hash, summary, tags,
                    char_count, chunk_count, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                document_to_database_tuple(document),
            )
        connection.execute(
            "INSERT OR REPLACE INTO app_meta (key, value) VALUES (?, ?)",
            ("documents_json_migrated", datetime.now(timezone.utc).isoformat()),
        )
        connection.commit()


def mark_documents_json_migrated() -> None:
    with open_database() as connection:
        connection.execute(
            "INSERT OR REPLACE INTO app_meta (key, value) VALUES (?, ?)",
            ("documents_json_migrated", datetime.now(timezone.utc).isoformat()),
        )
        connection.commit()


def normalize_document_tags(tags: list[str]) -> list[str]:
    normalized_tags: list[str] = []
    seen: set[str] = set()

    for raw_tag in tags:
        tag = re.sub(r"\s+", " ", str(raw_tag)).strip()
        if not tag or len(tag) > DOCUMENT_TAG_MAX_LENGTH or tag in seen:
            continue
        normalized_tags.append(tag)
        seen.add(tag)
        if len(normalized_tags) >= DOCUMENT_TAG_MAX_COUNT:
            break

    return normalized_tags


def document_to_database_tuple(
    document: DocumentRecord,
) -> tuple[str, str, str, str, str, str, str, str, str, int, int, str]:
    return (
        document.id,
        document.filename,
        document.file_type,
        document.stored_path,
        document.parsed_path,
        document.chunks_path,
        document.content_hash,
        document.summary,
        json.dumps(normalize_document_tags(document.tags), ensure_ascii=False),
        document.char_count,
        document.chunk_count,
        document.created_at,
    )


def document_from_row(row: sqlite3.Row) -> DocumentRecord:
    raw_tags = str(row["tags"] or "[]") if "tags" in row.keys() else "[]"
    try:
        tags = json.loads(raw_tags)
    except json.JSONDecodeError:
        tags = []

    return DocumentRecord(
        id=str(row["id"]),
        filename=str(row["filename"]),
        file_type=str(row["file_type"]),
        stored_path=str(row["stored_path"]),
        parsed_path=str(row["parsed_path"] or ""),
        chunks_path=str(row["chunks_path"] or ""),
        content_hash=str(row["content_hash"] or "") if "content_hash" in row.keys() else "",
        summary=str(row["summary"] or "") if "summary" in row.keys() else "",
        tags=normalize_document_tags(tags if isinstance(tags, list) else []),
        char_count=int(row["char_count"] or 0),
        chunk_count=int(row["chunk_count"] or 0),
        created_at=str(row["created_at"]),
    )


def load_documents() -> list[DocumentRecord]:
    ensure_data_dirs()

    with open_database() as connection:
        rows = connection.execute(
            """
            SELECT id, filename, file_type, stored_path, parsed_path,
                   chunks_path, content_hash, summary, tags,
                   char_count, chunk_count, created_at
            FROM documents
            ORDER BY created_at DESC
            """
        ).fetchall()

    return [document_from_row(row) for row in rows]


def insert_document_record(document: DocumentRecord) -> None:
    ensure_data_dirs()
    with open_database() as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO documents (
                id, filename, file_type, stored_path, parsed_path,
                chunks_path, content_hash, summary, tags,
                char_count, chunk_count, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            document_to_database_tuple(document),
        )
        connection.commit()


def delete_document_record(document_id: str) -> None:
    ensure_data_dirs()
    with open_database() as connection:
        connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        connection.commit()


def save_documents(documents: list[DocumentRecord]) -> None:
    ensure_data_dirs()
    with open_database() as connection:
        connection.execute("DELETE FROM documents")
        connection.executemany(
            """
            INSERT OR REPLACE INTO documents (
                id, filename, file_type, stored_path, parsed_path,
                chunks_path, content_hash, summary, tags,
                char_count, chunk_count, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [document_to_database_tuple(document) for document in documents],
        )
        connection.commit()


def sorted_documents() -> list[DocumentRecord]:
    return sorted(load_documents(), key=lambda document: document.created_at, reverse=True)


def build_document_list_response(page: int, page_size: int) -> DocumentListResponse:
    documents = sorted_documents()
    total = len(documents)
    total_pages = max(1, math.ceil(total / page_size))
    safe_page = min(page, total_pages)
    start = (safe_page - 1) * page_size
    end = start + page_size

    return DocumentListResponse(
        items=documents[start:end],
        total=total,
        page=safe_page,
        page_size=page_size,
        total_pages=total_pages,
        total_chunks=sum(document.chunk_count for document in documents),
    )


def create_ingest_task(task_id: str, total: int) -> None:
    ensure_data_dirs()
    now = datetime.now(timezone.utc).isoformat()
    with open_database() as connection:
        connection.execute(
            """
            INSERT INTO ingest_tasks (
                task_id, status, total, succeeded, failed,
                created_at, updated_at, completed_at
            )
            VALUES (?, ?, ?, 0, 0, ?, ?, '')
            """,
            (task_id, "queued", total, now, now),
        )
        connection.commit()


def create_ingest_task_file(
    *,
    file_id: str,
    task_id: str,
    document_id: str,
    filename: str,
    file_type: str,
    stored_path: str,
    status: IngestFileStatus,
    stage: IngestFileStage,
    error: str = "",
    duplicate_document_id: str = "",
) -> None:
    ensure_data_dirs()
    now = datetime.now(timezone.utc).isoformat()
    with open_database() as connection:
        connection.execute(
            """
            INSERT INTO ingest_task_files (
                file_id, task_id, document_id, filename, file_type,
                stored_path, duplicate_document_id, status, stage,
                error, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                file_id,
                task_id,
                document_id,
                filename,
                file_type,
                stored_path,
                duplicate_document_id,
                status,
                stage,
                error,
                now,
                now,
            ),
        )
        connection.commit()


def update_ingest_task_status(task_id: str, status: IngestTaskStatus) -> None:
    now = datetime.now(timezone.utc).isoformat()
    completed_at = now if status in {"completed", "failed", "partial_failed"} else ""
    with open_database() as connection:
        connection.execute(
            """
            UPDATE ingest_tasks
            SET status = ?, updated_at = ?, completed_at = CASE WHEN ? != '' THEN ? ELSE completed_at END
            WHERE task_id = ?
            """,
            (status, now, completed_at, completed_at, task_id),
        )
        connection.commit()


def update_ingest_task_file(
    file_id: str,
    *,
    status: IngestFileStatus | None = None,
    stage: IngestFileStage | None = None,
    char_count: int | None = None,
    chunk_count: int | None = None,
    error: str | None = None,
    duplicate_document_id: str | None = None,
) -> None:
    updates: list[str] = ["updated_at = ?"]
    values: list[object] = [datetime.now(timezone.utc).isoformat()]

    if status is not None:
        updates.append("status = ?")
        values.append(status)
    if stage is not None:
        updates.append("stage = ?")
        values.append(stage)
    if char_count is not None:
        updates.append("char_count = ?")
        values.append(char_count)
    if chunk_count is not None:
        updates.append("chunk_count = ?")
        values.append(chunk_count)
    if error is not None:
        updates.append("error = ?")
        values.append(error)
    if duplicate_document_id is not None:
        updates.append("duplicate_document_id = ?")
        values.append(duplicate_document_id)

    values.append(file_id)

    with open_database() as connection:
        connection.execute(
            f"UPDATE ingest_task_files SET {', '.join(updates)} WHERE file_id = ?",
            values,
        )
        connection.commit()


def refresh_ingest_task_counters(task_id: str) -> None:
    with open_database() as connection:
        task = connection.execute(
            "SELECT status FROM ingest_tasks WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        rows = connection.execute(
            "SELECT status FROM ingest_task_files WHERE task_id = ?",
            (task_id,),
        ).fetchall()

        total = len(rows)
        succeeded = sum(1 for row in rows if row["status"] == "indexed")
        failed = sum(1 for row in rows if row["status"] in {"failed", "duplicate"})
        running = any(row["status"] == "running" for row in rows)
        queued = any(row["status"] == "queued" for row in rows)

        current_task_status = str(task["status"]) if task else ""

        if running or (current_task_status == "running" and queued):
            status = "running"
        elif queued:
            status = "queued"
        elif failed and succeeded:
            status = "partial_failed"
        elif failed and not succeeded:
            status = "failed"
        else:
            status = "completed"

        now = datetime.now(timezone.utc).isoformat()
        completed_at = now if status in {"completed", "failed", "partial_failed"} else ""
        connection.execute(
            """
            UPDATE ingest_tasks
            SET status = ?, total = ?, succeeded = ?, failed = ?,
                updated_at = ?, completed_at = CASE WHEN ? != '' THEN ? ELSE completed_at END
            WHERE task_id = ?
            """,
            (status, total, succeeded, failed, now, completed_at, completed_at, task_id),
        )
        connection.commit()


def get_document_or_none(document_id: str) -> DocumentRecord | None:
    if not document_id:
        return None

    with open_database() as connection:
        row = connection.execute(
            """
            SELECT id, filename, file_type, stored_path, parsed_path,
                   chunks_path, content_hash, summary, tags,
                   char_count, chunk_count, created_at
            FROM documents
            WHERE id = ?
            """,
            (document_id,),
        ).fetchone()

    return document_from_row(row) if row else None


def build_ingest_task_response(task_id: str) -> IngestTaskResponse:
    ensure_data_dirs()
    refresh_ingest_task_counters(task_id)

    with open_database() as connection:
        task = connection.execute(
            """
            SELECT task_id, status, total, succeeded, failed,
                   created_at, updated_at, completed_at
            FROM ingest_tasks
            WHERE task_id = ?
            """,
            (task_id,),
        ).fetchone()

        if task is None:
            raise HTTPException(status_code=404, detail="Ingest task not found")

        rows = connection.execute(
            """
            SELECT file_id, filename, status, stage, document_id,
                   duplicate_document_id, char_count, chunk_count, error
            FROM ingest_task_files
            WHERE task_id = ?
            ORDER BY created_at ASC
            """,
            (task_id,),
        ).fetchall()

    items = [
        IngestTaskFileResponse(
            file_id=str(row["file_id"]),
            filename=str(row["filename"]),
            status=row["status"],
            stage=row["stage"],
            document_id=str(row["document_id"] or ""),
            document=get_document_or_none(str(row["document_id"] or "")),
            duplicate_document=get_document_or_none(str(row["duplicate_document_id"] or "")),
            char_count=int(row["char_count"] or 0),
            chunk_count=int(row["chunk_count"] or 0),
            error=str(row["error"] or ""),
        )
        for row in rows
    ]

    return IngestTaskResponse(
        task_id=str(task["task_id"]),
        status=task["status"],
        total=int(task["total"] or 0),
        succeeded=int(task["succeeded"] or 0),
        failed=int(task["failed"] or 0),
        created_at=str(task["created_at"]),
        updated_at=str(task["updated_at"]),
        completed_at=str(task["completed_at"] or ""),
        items=items,
    )


def normalize_session_id(session_id: str | None) -> str:
    if session_id and re.match(r"^[A-Za-z0-9_-]{6,80}$", session_id):
        return session_id
    return f"session_{uuid.uuid4().hex[:12]}"


def ensure_chat_session(session_id: str | None, first_question: str) -> str:
    ensure_data_dirs()
    normalized_session_id = normalize_session_id(session_id)
    now = datetime.now(timezone.utc).isoformat()
    title = first_question.strip()[:80] or "New conversation"

    with open_database() as connection:
        existing = connection.execute(
            "SELECT session_id FROM chat_sessions WHERE session_id = ?",
            (normalized_session_id,),
        ).fetchone()
        if existing is None:
            connection.execute(
                """
                INSERT INTO chat_sessions (session_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (normalized_session_id, title, now, now),
            )
        else:
            connection.execute(
                "UPDATE chat_sessions SET updated_at = ? WHERE session_id = ?",
                (now, normalized_session_id),
            )
        connection.commit()

    return normalized_session_id


def chat_message_from_row(row: sqlite3.Row) -> ChatMessage:
    return ChatMessage(
        id=str(row["id"]),
        session_id=str(row["session_id"]),
        role=row["role"],
        content=str(row["content"]),
        created_at=str(row["created_at"]),
    )


def load_chat_messages(session_id: str, limit: int = 12) -> list[ChatMessage]:
    ensure_data_dirs()
    with open_database() as connection:
        rows = connection.execute(
            """
            SELECT id, session_id, role, content, created_at
            FROM chat_messages
            WHERE session_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()

    return [chat_message_from_row(row) for row in reversed(rows)]


def add_chat_message(session_id: str, role: Literal["user", "assistant"], content: str) -> ChatMessage:
    ensure_data_dirs()
    now = datetime.now(timezone.utc).isoformat()
    message = ChatMessage(
        id=f"msg_{uuid.uuid4().hex[:12]}",
        session_id=session_id,
        role=role,
        content=content,
        created_at=now,
    )

    with open_database() as connection:
        connection.execute(
            """
            INSERT INTO chat_messages (id, session_id, role, content, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (message.id, message.session_id, message.role, message.content, message.created_at),
        )
        connection.execute(
            "UPDATE chat_sessions SET updated_at = ? WHERE session_id = ?",
            (now, session_id),
        )
        connection.commit()

    return message


def build_contextual_question(question: str, history: list[ChatMessage]) -> str:
    recent_history = history[-6:]
    if not recent_history:
        return question

    history_lines = [
        f"{'用户' if message.role == 'user' else '助手'}：{message.content[:260]}"
        for message in recent_history
    ]
    return (
        "以下是当前会话的最近上下文，请将最后的问题理解为可独立检索的问题。\n"
        + "\n".join(history_lines)
        + f"\n当前问题：{question}"
    )


def compact_preview_text(text: str, max_chars: int = OVERVIEW_PREVIEW_CHARS) -> str:
    compacted = re.sub(r"\s+", " ", text).strip()
    if len(compacted) <= max_chars:
        return compacted
    return compacted[:max_chars].rstrip() + "..."


def is_knowledge_overview_question(question: str) -> bool:
    normalized = re.sub(r"\s+", "", question).lower()
    if not normalized:
        return False

    scope_terms = [
        "知识库",
        "文档库",
        "资料库",
        "已上传",
        "上传了",
        "上传的",
        "当前资料",
        "当前文档",
        "库里",
    ]
    overview_terms = [
        "有哪些",
        "有什么",
        "包含什么",
        "包含哪些",
        "收录",
        "目录",
        "清单",
        "列表",
        "概览",
        "范围",
        "多少",
        "几份",
        "哪些文件",
        "哪些文档",
        "哪些内容",
    ]
    direct_questions = [
        "当前知识库有哪些内容",
        "知识库有哪些内容",
        "当前知识库有什么",
        "文档库有哪些内容",
        "我上传了哪些文件",
        "我上传了哪些文档",
    ]

    if any(item in normalized for item in direct_questions):
        return True

    return any(term in normalized for term in scope_terms) and any(
        term in normalized for term in overview_terms
    )


def build_document_overview_item(document: DocumentRecord) -> KnowledgeOverviewDocument:
    preview = compact_preview_text(document.summary, OVERVIEW_PREVIEW_CHARS)
    if not preview:
        try:
            chunks = load_chunks(document)
        except Exception:
            chunks = []
    else:
        chunks = []

    if not preview and chunks:
        first_chunk = next((chunk for chunk in chunks if chunk.content.strip()), chunks[0])
        heading = " / ".join(first_chunk.section_path) or first_chunk.title
        content_preview = compact_preview_text(first_chunk.content)
        preview = f"{heading}：{content_preview}" if heading else content_preview
    elif not preview:
        parsed_text = load_text_preview(document)
        preview = compact_preview_text(parsed_text)

    return KnowledgeOverviewDocument(
        document_id=document.id,
        filename=document.filename,
        file_type=document.file_type,
        char_count=document.char_count,
        chunk_count=document.chunk_count,
        created_at=document.created_at,
        preview=preview,
        summary=document.summary,
        tags=document.tags,
    )


def build_knowledge_overview() -> KnowledgeOverview:
    documents = sorted_documents()
    overview_documents = [
        build_document_overview_item(document)
        for document in documents[:OVERVIEW_MAX_DOCUMENTS_IN_ANSWER]
    ]

    return KnowledgeOverview(
        document_count=len(documents),
        total_chunks=sum(document.chunk_count for document in documents),
        total_char_count=sum(document.char_count for document in documents),
        documents=overview_documents,
        truncated=len(documents) > OVERVIEW_MAX_DOCUMENTS_IN_ANSWER,
    )


def build_knowledge_overview_answer(overview: KnowledgeOverview) -> str:
    if overview.document_count == 0:
        return (
            "当前知识库还没有文档。请先上传 txt、md 或 pdf 文件，系统完成解析、"
            "chunking、embedding 和入库后，我就可以回答知识库范围内的问题。"
        )

    lines = [
        (
            f"当前知识库共有 {overview.document_count} 份文档，"
            f"约 {overview.total_char_count} 个字符，"
            f"已切分为 {overview.total_chunks} 个 chunks。"
        ),
        "",
        "文档概览：",
    ]

    for index, document in enumerate(overview.documents, start=1):
        lines.append(
            (
                f"{index}. 《{document.filename}》"
                f"（{document.file_type}，{document.char_count} 字符，"
                f"{document.chunk_count} 个 chunks）"
            )
        )
        if document.preview:
            lines.append(f"   内容线索：{document.preview}")
        if document.tags:
            lines.append(f"   标签：{'、'.join(document.tags)}")

    if overview.truncated:
        hidden_count = overview.document_count - len(overview.documents)
        lines.append("")
        lines.append(f"还有 {hidden_count} 份文档未在本次回答中展开，可在文档库分页中继续查看。")

    lines.extend(
        [
            "",
            "你可以继续问：",
            "- 基于某一份文档总结核心观点",
            "- 对比几份文档的共同主题",
            "- 只围绕某个文件或章节提问",
        ]
    )
    return "\n".join(lines)


def build_knowledge_overview_sources(overview: KnowledgeOverview) -> list[Source]:
    return [
        Source(
            title=document.filename,
            content=document.preview or "该文档暂无可展示的内容线索。",
            document_id=document.document_id,
            score=None,
        )
        for document in overview.documents
    ]


def repair_mojibake_filename(filename: str) -> str:
    if not filename:
        return "uploaded-document"

    try:
        repaired = filename.encode("latin-1").decode("utf-8")
    except UnicodeError:
        return filename

    return repaired or filename


def safe_filename(filename: str) -> str:
    name = Path(filename).name.strip()
    return name or "uploaded-document"


def relative_path(path: Path) -> str:
    return str(path.relative_to(BASE_DIR)).replace("\\", "/")


def calculate_file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_duplicate_document(
    filename: str,
    content_hash: str = "",
) -> tuple[DocumentRecord | None, str]:
    ensure_data_dirs()
    with open_database() as connection:
        row = None
        reason = ""

        if content_hash:
            row = connection.execute(
                """
                SELECT id, filename, file_type, stored_path, parsed_path,
                       chunks_path, content_hash, summary, tags,
                       char_count, chunk_count, created_at
                FROM documents
                WHERE content_hash = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (content_hash,),
            ).fetchone()
            if row is not None:
                reason = "content_hash"

        if row is None:
            row = connection.execute(
                """
                SELECT id, filename, file_type, stored_path, parsed_path,
                       chunks_path, content_hash, summary, tags,
                       char_count, chunk_count, created_at
                FROM documents
                WHERE filename = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (filename,),
            ).fetchone()
            if row is not None:
                reason = "filename"

    return (document_from_row(row), reason) if row is not None else (None, "")


def duplicate_upload_message(filename: str, duplicate: DocumentRecord, reason: str) -> str:
    if reason == "content_hash":
        return f"重复上传：文件内容与已入库文档《{duplicate.filename}》完全一致。"
    return f"重复上传：已存在同名文档《{filename}》。"


def generate_document_summary(
    filename: str,
    parsed_text: str,
    chunks: list[DocumentChunk],
) -> str:
    section_titles: list[str] = []
    seen_titles: set[str] = set()

    for chunk in chunks:
        candidates = [item for item in chunk.section_path if item.strip()]
        if chunk.title:
            candidates.append(chunk.title)

        for candidate in candidates:
            title = compact_preview_text(candidate, 42)
            if title and title not in seen_titles:
                section_titles.append(title)
                seen_titles.add(title)
                break

        if len(section_titles) >= 4:
            break

    preview = compact_preview_text(parsed_text, DOCUMENT_SUMMARY_CHARS)
    if section_titles and preview:
        return f"主题线索：{'、'.join(section_titles)}。内容摘要：{preview}"
    if preview:
        return f"内容摘要：{preview}"
    return f"《{filename}》已入库，但解析文本较短，暂时无法生成更详细摘要。"


def update_document_tags(document_id: str, tags: list[str]) -> DocumentRecord:
    normalized_tags = normalize_document_tags(tags)
    ensure_data_dirs()
    with open_database() as connection:
        connection.execute(
            "UPDATE documents SET tags = ? WHERE id = ?",
            (json.dumps(normalized_tags, ensure_ascii=False), document_id),
        )
        connection.commit()

    return find_document(document_id)


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


def parse_text_file(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue

    return path.read_text(encoding="utf-8", errors="ignore")


def parse_pdf_file(path: Path) -> str:
    reader = PdfReader(str(path))
    pages: list[str] = []

    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"[Page {index}]\n{text.strip()}")

    return "\n\n".join(pages)


def normalize_extracted_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def parse_document(path: Path, extension: str) -> str:
    if extension in {".txt", ".md"}:
        return parse_text_file(path)

    if extension == ".pdf":
        return parse_pdf_file(path)

    raise HTTPException(status_code=400, detail="Unsupported file type")


def detect_heading(line: str) -> Heading | None:
    text = line.strip()
    if not text or len(text) > 120:
        return None

    markdown = re.match(r"^(#{1,6})\s+(.+)$", text)
    if markdown:
        return Heading(title=markdown.group(2).strip(), level=len(markdown.group(1)))

    numbered = re.match(r"^(\d+(?:\.\d+){0,5})[.\s]+(.+)$", text)
    if numbered:
        return Heading(title=text, level=min(numbered.group(1).count(".") + 1, 6))

    chinese_number = r"[一二三四五六七八九十百]+"
    if re.match(rf"^{chinese_number}[、.．\s]+(.+)$", text):
        return Heading(title=text, level=1)

    if re.match(rf"^[（(]{chinese_number}[）)]\s*(.+)$", text):
        return Heading(title=text, level=2)

    if re.match(rf"^第\s*({chinese_number}|\d+)\s*[章节篇部分]\s+(.+)$", text):
        return Heading(title=text, level=1)

    if re.match(r"^(chapter|section)\s+\d+[:.\s]+(.+)$", text, re.IGNORECASE):
        return Heading(title=text, level=1)

    return None


def split_sentences(text: str) -> list[str]:
    pieces = re.split(r"(?<=[。！？!?])\s*", text)
    return [piece.strip() for piece in pieces if piece.strip()]


def build_semantic_units(text: str) -> list[SemanticUnit]:
    units: list[SemanticUnit] = []
    current_title = ""
    current_level: int | None = None
    paragraphs = [item.strip() for item in re.split(r"\n\s*\n", text.strip()) if item.strip()]

    if len(paragraphs) <= 1:
        paragraphs = split_sentences(text)

    for paragraph in paragraphs:
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        if not lines:
            continue

        if len(lines) == 1:
            heading = detect_heading(lines[0])
            if heading:
                current_title = heading.title
                current_level = heading.level
                units.append(
                    SemanticUnit(
                        content=lines[0],
                        title=current_title,
                        heading_level=current_level,
                        is_heading=True,
                    )
                )
                continue

        content = "\n".join(lines)
        if len(content) > MAX_CHUNK_SIZE:
            for sentence in split_sentences(content):
                units.append(
                    SemanticUnit(
                        content=sentence,
                        title=current_title,
                        heading_level=current_level,
                    )
                )
        else:
            units.append(
                SemanticUnit(
                    content=content,
                    title=current_title,
                    heading_level=current_level,
                )
            )

    return units


def detect_heading(line: str) -> Heading | None:
    text = line.strip()
    if not text or len(text) > 120:
        return None

    markdown = re.match(r"^(#{1,6})\s+(.+)$", text)
    if markdown:
        return Heading(title=markdown.group(2).strip(), level=len(markdown.group(1)))

    numbered = re.match(r"^(\d+(?:\.\d+){0,5})[.\s]+(.+)$", text)
    if numbered:
        return Heading(title=text, level=min(numbered.group(1).count(".") + 1, 6))

    chinese_number = r"[一二三四五六七八九十百]+"
    if re.match(rf"^{chinese_number}[、.．\s]+(.+)$", text):
        return Heading(title=text, level=1)

    if re.match(rf"^[（(]{chinese_number}[）)]\s*(.+)$", text):
        return Heading(title=text, level=2)

    if re.match(rf"^第\s*({chinese_number}|\d+)\s*[章节篇部分]\s+(.+)$", text):
        return Heading(title=text, level=1)

    if re.match(r"^(chapter|section)\s+\d+[:.\s]+(.+)$", text, re.IGNORECASE):
        return Heading(title=text, level=1)

    return None


def split_sentences(text: str) -> list[str]:
    pieces = re.split(r"(?<=[。！？!?])\s*", text)
    return [piece.strip() for piece in pieces if piece.strip()]


def estimate_tokens(text: str) -> int:
    latin_words = re.findall(r"[A-Za-z0-9]+", text)
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    other_chars = max(len(text) - sum(len(word) for word in latin_words) - len(chinese_chars), 0)
    return len(latin_words) + math.ceil(len(chinese_chars) / 1.6) + math.ceil(other_chars / 4)


def update_section_path(section_path: list[str], heading: Heading) -> list[str]:
    level = max(1, min(heading.level, 6))
    next_path = section_path[: level - 1]
    next_path.append(heading.title)
    return next_path


def split_text_into_page_blocks(text: str) -> list[tuple[int | None, str]]:
    normalized = normalize_extracted_text(text)
    marker_pattern = re.compile(r"^\[Page\s+(\d+)\]\s*$", re.MULTILINE)
    matches = list(marker_pattern.finditer(normalized))

    if not matches:
        return [(None, normalized)]

    blocks: list[tuple[int | None, str]] = []
    for index, match in enumerate(matches):
        page_number = int(match.group(1))
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized)
        page_text = normalized[start:end].strip()
        if page_text:
            blocks.append((page_number, page_text))

    return blocks


def build_semantic_units(text: str) -> list[SemanticUnit]:
    units: list[SemanticUnit] = []
    current_title = ""
    current_level: int | None = None
    section_path: list[str] = []

    for page_number, page_text in split_text_into_page_blocks(text):
        paragraphs = [
            item.strip()
            for item in re.split(r"\n\s*\n", page_text.strip())
            if item.strip()
        ]

        if len(paragraphs) <= 1:
            paragraphs = split_sentences(page_text)

        for paragraph in paragraphs:
            lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
            if not lines:
                continue

            if len(lines) == 1:
                heading = detect_heading(lines[0])
                if heading:
                    current_title = heading.title
                    current_level = heading.level
                    section_path = update_section_path(section_path, heading)
                    units.append(
                        SemanticUnit(
                            content=lines[0],
                            title=current_title,
                            heading_level=current_level,
                            is_heading=True,
                            section_path=section_path.copy(),
                            page_start=page_number,
                            page_end=page_number,
                        )
                    )
                    continue

            content = "\n".join(lines)
            if len(content) > MAX_CHUNK_SIZE:
                for sentence in split_sentences(content):
                    units.append(
                        SemanticUnit(
                            content=sentence,
                            title=current_title,
                            heading_level=current_level,
                            section_path=section_path.copy(),
                            page_start=page_number,
                            page_end=page_number,
                        )
                    )
            else:
                units.append(
                    SemanticUnit(
                        content=content,
                        title=current_title,
                        heading_level=current_level,
                        section_path=section_path.copy(),
                        page_start=page_number,
                        page_end=page_number,
                    )
                )

    return units


def cosine_similarity(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def normalize_search_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def extract_search_terms(text: str) -> list[str]:
    normalized = normalize_search_text(text)
    latin_terms = re.findall(r"[a-z0-9][a-z0-9_+#.-]{1,}", normalized)
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", normalized)
    chinese_bigrams = [
        f"{left}{right}" for left, right in zip(chinese_chars, chinese_chars[1:])
    ]
    chinese_trigrams = [
        "".join(chinese_chars[index : index + 3])
        for index in range(max(len(chinese_chars) - 2, 0))
    ]

    terms: list[str] = []
    seen: set[str] = set()
    for term in latin_terms + chinese_bigrams + chinese_trigrams:
        if len(term) < 2 or term in seen:
            continue
        seen.add(term)
        terms.append(term)

    return terms[:80]


def lexical_rerank_score(question: str, result: SearchResult, query_terms: list[str]) -> float:
    if not query_terms:
        return 0.0

    content = normalize_search_text(result.content)
    title = normalize_search_text(result.title)
    section = normalize_search_text(" ".join(result.section_path))
    filename = normalize_search_text(result.document_filename)
    searchable = f"{filename} {title} {section} {content}"

    weighted_hits = 0.0
    possible_weight = 0.0
    for term in query_terms:
        term_weight = 1.0 + min(len(term), 8) / 8
        possible_weight += term_weight
        if term in searchable:
            weighted_hits += term_weight
            if term in title:
                weighted_hits += 0.35
            if term in section:
                weighted_hits += 0.25
            if term in filename:
                weighted_hits += 0.15

    coverage_score = weighted_hits / max(possible_weight, 1.0)

    phrase_score = 0.0
    normalized_question = normalize_search_text(question)
    if len(normalized_question) >= 4 and normalized_question in searchable:
        phrase_score = 1.0

    return round(max(0.0, min(1.0, coverage_score * 0.82 + phrase_score * 0.18)), 6)


def apply_hybrid_rerank(
    *, question: str, results: list[SearchResult], query_terms: list[str]
) -> list[SearchResult]:
    reranked: list[SearchResult] = []

    for result in results:
        vector_score = result.vector_score or result.score
        lexical_score = lexical_rerank_score(question, result, query_terms)
        structural_boost = 0.0
        if result.title:
            structural_boost += 0.025
        if result.section_path:
            structural_boost += 0.025

        rerank_score = min(
            1.0,
            max(0.0, vector_score * 0.72 + lexical_score * 0.28 + structural_boost),
        )
        result.vector_score = round(vector_score, 6)
        result.lexical_score = lexical_score
        result.rerank_score = round(rerank_score, 6)
        result.score = result.rerank_score
        reranked.append(result)

    return sorted(
        reranked,
        key=lambda item: (item.rerank_score, item.lexical_score, item.vector_score),
        reverse=True,
    )


def semantic_threshold(similarities: list[float]) -> float:
    if not similarities:
        return 0.0

    mean = sum(similarities) / len(similarities)
    variance = sum((item - mean) ** 2 for item in similarities) / len(similarities)
    std = math.sqrt(variance)
    return max(0.18, min(0.62, mean - 0.35 * std))


def split_by_length(text: str) -> list[str]:
    normalized = text.strip()
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(normalized):
        end = min(start + TARGET_CHUNK_SIZE, len(normalized))
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(normalized):
            break

        start = max(end - 150, start + 1)

    return chunks


def build_chunk(
    document_id: str,
    index: int,
    content: str,
    title: str = "",
    heading_level: int | None = None,
    strategy: ChunkStrategy = "length_fallback",
    semantic_break_score: float | None = None,
    section_path: list[str] | None = None,
    page_start: int | None = None,
    page_end: int | None = None,
) -> DocumentChunk:
    clean_content = content.strip()
    return DocumentChunk(
        id=f"{document_id}_chunk_{index:04d}",
        document_id=document_id,
        index=index,
        content=clean_content,
        char_count=len(clean_content),
        title=title,
        heading_level=heading_level,
        strategy=strategy,
        semantic_break_score=semantic_break_score,
        section_path=section_path or [],
        token_estimate=estimate_tokens(clean_content),
        page_start=page_start,
        page_end=page_end,
        embedding=embed_text(clean_content),
        embedding_provider=embedding_provider(),
    )


def add_chunk(
    chunks: list[DocumentChunk],
    document_id: str,
    content: str,
    title: str,
    heading_level: int | None,
    strategy: ChunkStrategy,
    semantic_break_score: float | None = None,
    section_path: list[str] | None = None,
    page_start: int | None = None,
    page_end: int | None = None,
) -> None:
    clean_content = content.strip()
    if not clean_content:
        return

    if len(clean_content) > MAX_CHUNK_SIZE:
        for part in split_by_length(clean_content):
            add_chunk(
                chunks=chunks,
                document_id=document_id,
                content=part,
                title=title,
                heading_level=heading_level,
                strategy="section_semantic_split",
                semantic_break_score=semantic_break_score,
                section_path=section_path,
                page_start=page_start,
                page_end=page_end,
            )
        return

    chunks.append(
        build_chunk(
            document_id=document_id,
            index=len(chunks),
            content=clean_content,
            title=title,
            heading_level=heading_level,
            strategy=strategy,
            semantic_break_score=semantic_break_score,
            section_path=section_path,
            page_start=page_start,
            page_end=page_end,
        )
    )


def split_text_into_chunks(text: str, document_id: str) -> list[DocumentChunk]:
    normalized_text = normalize_extracted_text(text)
    if not normalized_text:
        return []

    units = build_semantic_units(normalized_text)
    if len(units) <= 1:
        return [
            build_chunk(
                document_id=document_id,
                index=index,
                content=content,
                strategy="length_fallback",
            )
            for index, content in enumerate(split_by_length(normalized_text))
        ]

    embeddings = [embed_text(unit.content) for unit in units]
    similarities = [
        cosine_similarity(embeddings[index - 1], embeddings[index])
        for index in range(1, len(embeddings))
    ]
    threshold = semantic_threshold(similarities)

    chunks: list[DocumentChunk] = []
    current_units: list[SemanticUnit] = []
    current_size = 0
    last_break_score: float | None = None

    for index, unit in enumerate(units):
        previous_similarity = similarities[index - 1] if index > 0 else None
        semantic_break = (
            previous_similarity is not None
            and previous_similarity <= threshold
            and current_size >= MIN_CHUNK_SIZE
        )
        heading_break = unit.is_heading and current_size >= MIN_HEADING_CHUNK_SIZE
        size_break = current_size + len(unit.content) > MAX_CHUNK_SIZE

        if current_units and (semantic_break or heading_break or size_break):
            title = next((item.title for item in reversed(current_units) if item.title), "")
            heading_level = next(
                (item.heading_level for item in reversed(current_units) if item.heading_level),
                None,
            )
            section_path = next(
                (item.section_path for item in reversed(current_units) if item.section_path),
                [],
            )
            pages = [item.page_start for item in current_units if item.page_start is not None]
            add_chunk(
                chunks=chunks,
                document_id=document_id,
                content="\n\n".join(item.content for item in current_units),
                title=title,
                heading_level=heading_level,
                strategy="section_semantic" if not size_break else "section_semantic_split",
                semantic_break_score=last_break_score,
                section_path=section_path,
                page_start=min(pages) if pages else None,
                page_end=max(pages) if pages else None,
            )
            current_units = []
            current_size = 0

        current_units.append(unit)
        current_size += len(unit.content)
        last_break_score = previous_similarity

    if current_units:
        title = next((item.title for item in reversed(current_units) if item.title), "")
        heading_level = next(
            (item.heading_level for item in reversed(current_units) if item.heading_level),
            None,
        )
        section_path = next(
            (item.section_path for item in reversed(current_units) if item.section_path),
            [],
        )
        pages = [item.page_start for item in current_units if item.page_start is not None]
        add_chunk(
            chunks=chunks,
            document_id=document_id,
            content="\n\n".join(item.content for item in current_units),
            title=title,
            heading_level=heading_level,
            strategy="section_semantic",
            semantic_break_score=last_break_score,
            section_path=section_path,
            page_start=min(pages) if pages else None,
            page_end=max(pages) if pages else None,
        )

    apply_chunk_overlaps(chunks)
    return chunks


def apply_chunk_overlaps(chunks: list[DocumentChunk]) -> None:
    for index, chunk in enumerate(chunks):
        previous_chunk = chunks[index - 1] if index > 0 else None
        next_chunk = chunks[index + 1] if index + 1 < len(chunks) else None

        chunk.overlap_previous = (
            previous_chunk.content[-CHUNK_OVERLAP_CHARS:].strip()
            if previous_chunk
            else ""
        )
        chunk.overlap_next = (
            next_chunk.content[:CHUNK_OVERLAP_CHARS].strip()
            if next_chunk
            else ""
        )
        embedding_text = "\n".join(
            item
            for item in (chunk.overlap_previous, chunk.content, chunk.overlap_next)
            if item
        )
        chunk.embedding = embed_text(embedding_text)
        chunk.embedding_provider = embedding_provider()


def save_parsed_text(document_id: str, text: str) -> Path:
    parsed_path = PARSED_DIR / f"{document_id}.txt"
    parsed_path.write_text(text, encoding="utf-8")
    return parsed_path


def save_chunks(document_id: str, chunks: list[DocumentChunk]) -> Path:
    chunks_path = CHUNKS_DIR / f"{document_id}.json"
    payload = [chunk.model_dump() for chunk in chunks]
    chunks_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return chunks_path


def load_chunks(document: DocumentRecord) -> list[DocumentChunk]:
    if not document.chunks_path:
        return []

    chunks_path = BASE_DIR / document.chunks_path
    if not chunks_path.exists():
        return []

    try:
        raw_chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []

    chunks: list[DocumentChunk] = []
    changed = False
    for item in raw_chunks:
        chunk = DocumentChunk(**item)
        if chunk.token_estimate == 0:
            chunk.token_estimate = estimate_tokens(chunk.content)
        if not chunk.embedding or chunk.embedding_provider != embedding_provider():
            chunk.embedding = embed_text(chunk.content)
            chunk.embedding_provider = embedding_provider()
            changed = True
        chunks.append(chunk)

    apply_chunk_overlaps(chunks)
    changed = True

    if changed:
        save_chunks(document.id, chunks)

    return chunks


def load_parsed_preview(document: DocumentRecord) -> str:
    if not document.parsed_path:
        return ""

    parsed_path = BASE_DIR / document.parsed_path
    if not parsed_path.exists():
        return ""

    return parsed_path.read_text(encoding="utf-8", errors="ignore")[:DETAIL_PREVIEW_CHARS]


def find_document(document_id: str) -> DocumentRecord:
    for document in load_documents():
        if document.id == document_id:
            return document

    raise HTTPException(status_code=404, detail="Document not found")


def delete_data_file(relative_file_path: str) -> str | None:
    if not relative_file_path:
        return None

    file_path = (BASE_DIR / relative_file_path).resolve()
    data_root = DATA_DIR.resolve()

    try:
        file_path.relative_to(data_root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid document file path") from exc

    if file_path.exists() and file_path.is_file():
        file_path.unlink()
        return relative_path(file_path)

    return None


def search_chunks(request: SearchRequest) -> SearchResponse:
    question_embedding = embed_text(request.question)
    query_terms = extract_search_terms(request.question)
    candidate_top_k = min(
        RERANK_CANDIDATE_MAX,
        max(request.top_k, request.top_k * RERANK_CANDIDATE_MULTIPLIER),
    )

    if chroma_available():
        try:
            raw_chroma_results = query_chunks(question_embedding, candidate_top_k)
            if raw_chroma_results:
                candidate_results = [
                    SearchResult(
                        document_id=str(item["metadata"].get("document_id") or ""),
                        document_filename=str(
                            item["metadata"].get("document_filename") or ""
                        ),
                        chunk_id=str(item["metadata"].get("chunk_id") or item["chunk_id"]),
                        chunk_index=int(item["metadata"].get("chunk_index") or 0),
                        title=str(item["metadata"].get("title") or ""),
                        score=float(item["score"]),
                        vector_score=float(item["score"]),
                        content=str(item.get("content") or ""),
                        char_count=int(item["metadata"].get("char_count") or 0),
                        strategy=item["metadata"].get("strategy") or "length_fallback",
                        section_path=[
                            part.strip()
                            for part in str(
                                item["metadata"].get("section_path") or ""
                            ).split("/")
                            if part.strip()
                        ],
                        token_estimate=int(item["metadata"].get("token_estimate") or 0),
                        page_start=(
                            int(item["metadata"]["page_start"])
                            if item["metadata"].get("page_start") is not None
                            else None
                        ),
                        page_end=(
                            int(item["metadata"]["page_end"])
                            if item["metadata"].get("page_end") is not None
                            else None
                        ),
                    )
                    for item in raw_chroma_results
                ]
                reranked_results = apply_hybrid_rerank(
                    question=request.question,
                    results=candidate_results,
                    query_terms=query_terms,
                )
                filtered_results = [
                    item
                    for item in reranked_results
                    if item.score >= request.score_threshold
                ]
                return SearchResponse(
                    question=request.question,
                    top_k=request.top_k,
                    score_threshold=request.score_threshold,
                    total_chunks=get_vector_store_status()["chunk_count"],
                    results=filtered_results[: request.top_k],
                    mode="chroma",
                    query_terms=query_terms[:24],
                )
        except Exception:
            pass

    documents = load_documents()
    results: list[SearchResult] = []
    total_chunks = 0

    for document in documents:
        chunks = load_chunks(document)
        total_chunks += len(chunks)

        for chunk in chunks:
            chunk_embedding = chunk.embedding or embed_text(chunk.content)
            score = cosine_similarity(question_embedding, chunk_embedding)
            results.append(
                SearchResult(
                    document_id=document.id,
                    document_filename=document.filename,
                    chunk_id=chunk.id,
                    chunk_index=chunk.index,
                    title=chunk.title,
                    score=round(score, 6),
                    vector_score=round(score, 6),
                    content=chunk.content,
                    char_count=chunk.char_count,
                    strategy=chunk.strategy,
                    section_path=chunk.section_path,
                    token_estimate=chunk.token_estimate,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                )
            )

    ranked_results = apply_hybrid_rerank(
        question=request.question,
        results=results,
        query_terms=query_terms,
    )
    filtered_results = [
        item for item in ranked_results if item.score >= request.score_threshold
    ]
    return SearchResponse(
        question=request.question,
        top_k=request.top_k,
        score_threshold=request.score_threshold,
        total_chunks=total_chunks,
        results=filtered_results[: request.top_k],
        mode="local_hash_embedding",
        query_terms=query_terms[:24],
    )


def build_sources(results: list[SearchResult]) -> list[Source]:
    return [
        Source(
            title=f"{item.document_filename} / chunk {item.chunk_index}",
            content=item.content,
            document_id=item.document_id,
            chunk_id=item.chunk_id,
            score=item.score,
            page_start=item.page_start,
            page_end=item.page_end,
            section_path=item.section_path,
        )
        for item in results
    ]


def build_retrieval_template_answer(question: str, results: list[SearchResult]) -> str:
    summary_lines = [
        f"{index}. 《{item.document_filename}》chunk {item.chunk_index}，相似度 {item.score:.3f}"
        for index, item in enumerate(results, start=1)
    ]
    return (
        "已根据你的问题完成本地向量检索。当前没有成功调用 LLM，"
        "所以先返回命中的文档片段作为回答依据：\n"
        + "\n".join(summary_lines)
        + f"\n\n用户问题：{question}"
    )


def build_rag_prompt(question: str, results: list[SearchResult]) -> str:
    context_blocks = []
    for index, item in enumerate(results, start=1):
        page_text = ""
        if item.page_start:
            page_text = f"页码：{item.page_start}"
            if item.page_end and item.page_end != item.page_start:
                page_text += f"-{item.page_end}"

        section_text = " / ".join(item.section_path) if item.section_path else "未识别章节"
        context_blocks.append(
            "\n".join(
                [
                    f"[来源 {index}]",
                    f"文档：{item.document_filename}",
                    f"chunk：{item.chunk_index}",
                    f"相似度：{item.score:.3f}",
                    f"章节：{section_text}",
                    page_text,
                    "内容：",
                    item.content,
                ]
            ).strip()
        )

    return (
        "你是一个严谨的中文知识库问答助手。请只根据给定资料回答用户问题。\n"
        "如果资料不足，请明确说明“当前文档中没有足够信息”。\n"
        "回答要清晰、具体、不要编造。最后用简短列表给出依据来源。\n\n"
        f"用户问题：\n{question}\n\n"
        "参考资料：\n"
        + "\n\n".join(context_blocks)
    )


def call_deepseek_chat(question: str, results: list[SearchResult]) -> str | None:
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
                "content": "你是一个可靠的 RAG 问答助手，只能基于用户提供的参考资料回答。",
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


def build_retrieval_template_answer(question: str, results: list[SearchResult]) -> str:
    summary_lines = [
        f"{index}. 《{item.document_filename}》chunk {item.chunk_index}，相似度 {item.score:.3f}"
        for index, item in enumerate(results, start=1)
    ]
    return (
        "已完成知识库检索，但没有成功调用 LLM。以下是可作为回答依据的命中文档片段：\n"
        + "\n".join(summary_lines)
        + f"\n\n用户问题：{question}"
    )


def build_rag_prompt(question: str, results: list[SearchResult]) -> str:
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


def call_deepseek_chat(question: str, results: list[SearchResult]) -> str | None:
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


async def ingest_upload_file(file: UploadFile) -> DocumentRecord:
    ensure_data_dirs()

    original_filename = safe_filename(repair_mojibake_filename(file.filename or ""))
    extension = Path(original_filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        await file.close()
        raise HTTPException(
            status_code=400,
            detail="Only .txt, .md, and .pdf files are supported.",
        )

    duplicate_by_filename, duplicate_reason = find_duplicate_document(original_filename)
    if duplicate_by_filename:
        await file.close()
        raise HTTPException(
            status_code=409,
            detail=duplicate_upload_message(
                original_filename,
                duplicate_by_filename,
                duplicate_reason,
            ),
        )

    document_id = f"doc_{uuid.uuid4().hex[:12]}"
    stored_path = UPLOAD_DIR / f"{document_id}{extension}"

    try:
        with stored_path.open("wb") as target:
            shutil.copyfileobj(file.file, target)
    finally:
        await file.close()

    content_hash = calculate_file_hash(stored_path)
    duplicate_by_hash, duplicate_reason = find_duplicate_document(original_filename, content_hash)
    if duplicate_by_hash:
        stored_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=409,
            detail=duplicate_upload_message(original_filename, duplicate_by_hash, duplicate_reason),
        )

    try:
        parsed_text = parse_document(stored_path, extension)
        chunks = split_text_into_chunks(parsed_text, document_id)
        parsed_path = save_parsed_text(document_id, parsed_text)
        chunks_path = save_chunks(document_id, chunks)
    except Exception as exc:
        stored_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {exc}") from exc

    document = DocumentRecord(
        id=document_id,
        filename=original_filename,
        file_type=extension,
        stored_path=relative_path(stored_path),
        parsed_path=relative_path(parsed_path),
        chunks_path=relative_path(chunks_path),
        content_hash=content_hash,
        summary=generate_document_summary(original_filename, parsed_text, chunks),
        tags=[],
        char_count=len(parsed_text),
        chunk_count=len(chunks),
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    try:
        upsert_document_chunks(document, chunks)
    except Exception:
        pass

    insert_document_record(document)

    return document


def process_ingest_task(task_id: str) -> None:
    update_ingest_task_status(task_id, "running")

    with open_database() as connection:
        rows = connection.execute(
            """
            SELECT file_id, document_id, filename, file_type, stored_path
            FROM ingest_task_files
            WHERE task_id = ? AND status = ?
            ORDER BY created_at ASC
            """,
            (task_id, "queued"),
        ).fetchall()

    for row in rows:
        file_id = str(row["file_id"])
        document_id = str(row["document_id"])
        filename = str(row["filename"])
        extension = str(row["file_type"])
        stored_path = BASE_DIR / str(row["stored_path"])

        try:
            update_ingest_task_file(file_id, status="running", stage="parsing")
            content_hash = calculate_file_hash(stored_path)
            duplicate_document, duplicate_reason = find_duplicate_document(filename, content_hash)
            if duplicate_document:
                stored_path.unlink(missing_ok=True)
                update_ingest_task_file(
                    file_id,
                    status="duplicate",
                    stage="duplicate",
                    error=duplicate_upload_message(filename, duplicate_document, duplicate_reason),
                    duplicate_document_id=duplicate_document.id,
                )
                refresh_ingest_task_counters(task_id)
                continue

            parsed_text = parse_document(stored_path, extension)

            update_ingest_task_file(file_id, stage="chunking")
            chunks = split_text_into_chunks(parsed_text, document_id)

            parsed_path = save_parsed_text(document_id, parsed_text)
            chunks_path = save_chunks(document_id, chunks)
            summary = generate_document_summary(filename, parsed_text, chunks)

            document = DocumentRecord(
                id=document_id,
                filename=filename,
                file_type=extension,
                stored_path=relative_path(stored_path),
                parsed_path=relative_path(parsed_path),
                chunks_path=relative_path(chunks_path),
                content_hash=content_hash,
                summary=summary,
                tags=[],
                char_count=len(parsed_text),
                chunk_count=len(chunks),
                created_at=datetime.now(timezone.utc).isoformat(),
            )

            update_ingest_task_file(
                file_id,
                stage="embedding",
                char_count=document.char_count,
                chunk_count=document.chunk_count,
            )

            update_ingest_task_file(file_id, stage="indexing")
            try:
                upsert_document_chunks(document, chunks)
            except Exception:
                pass

            insert_document_record(document)
            update_ingest_task_file(
                file_id,
                status="indexed",
                stage="indexed",
                char_count=document.char_count,
                chunk_count=document.chunk_count,
                error="",
            )
        except Exception as exc:
            stored_path.unlink(missing_ok=True)
            update_ingest_task_file(
                file_id,
                status="failed",
                stage="failed",
                error=str(exc),
            )
        finally:
            refresh_ingest_task_counters(task_id)

    refresh_ingest_task_counters(task_id)


@app.get("/")
def read_root() -> dict[str, str]:
    return {
        "name": "RAG Agent System MVP",
        "status": "running",
        "version": APP_VERSION,
        "docs": "http://localhost:8000/docs",
        "health": "http://localhost:8000/health",
    }


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="rag-agent-system",
        version=APP_VERSION,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post(
    "/api/documents/upload",
    response_model=DocumentRecord,
    dependencies=[Depends(require_admin)],
)
async def upload_document(file: UploadFile = File(...)) -> DocumentRecord:
    return await ingest_upload_file(file)


@app.post(
    "/api/documents/upload/batch",
    response_model=IngestTaskResponse,
    dependencies=[Depends(require_admin)],
)
async def upload_documents_batch(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
) -> IngestTaskResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")

    ensure_data_dirs()
    task_id = f"task_{uuid.uuid4().hex[:12]}"
    create_ingest_task(task_id, len(files))
    queued_count = 0
    queued_filenames: set[str] = set()
    queued_hashes: set[str] = set()

    for file in files:
        filename = safe_filename(repair_mojibake_filename(file.filename or ""))
        extension = Path(filename).suffix.lower()
        file_id = f"file_{uuid.uuid4().hex[:12]}"
        document_id = f"doc_{uuid.uuid4().hex[:12]}"

        if extension not in SUPPORTED_EXTENSIONS:
            await file.close()
            create_ingest_task_file(
                file_id=file_id,
                task_id=task_id,
                document_id="",
                filename=filename,
                file_type=extension,
                stored_path="",
                status="failed",
                stage="failed",
                error="Only .txt, .md, and .pdf files are supported.",
            )
            continue

        duplicate_by_filename, duplicate_reason = find_duplicate_document(filename)
        if duplicate_by_filename or filename in queued_filenames:
            await file.close()
            duplicate_message = (
                duplicate_upload_message(filename, duplicate_by_filename, duplicate_reason)
                if duplicate_by_filename
                else f"重复上传：本次批量上传中已经包含同名文件《{filename}》。"
            )
            create_ingest_task_file(
                file_id=file_id,
                task_id=task_id,
                document_id="",
                filename=filename,
                file_type=extension,
                stored_path="",
                status="duplicate",
                stage="duplicate",
                error=duplicate_message,
                duplicate_document_id=duplicate_by_filename.id if duplicate_by_filename else "",
            )
            continue

        stored_path = UPLOAD_DIR / f"{document_id}{extension}"

        try:
            with stored_path.open("wb") as target:
                shutil.copyfileobj(file.file, target)
        except Exception as exc:
            create_ingest_task_file(
                file_id=file_id,
                task_id=task_id,
                document_id="",
                filename=filename,
                file_type=extension,
                stored_path="",
                status="failed",
                stage="failed",
                error=str(exc),
            )
        else:
            content_hash = calculate_file_hash(stored_path)
            duplicate_by_hash, duplicate_reason = find_duplicate_document(filename, content_hash)
            if duplicate_by_hash or content_hash in queued_hashes:
                stored_path.unlink(missing_ok=True)
                duplicate_message = (
                    duplicate_upload_message(filename, duplicate_by_hash, duplicate_reason)
                    if duplicate_by_hash
                    else f"重复上传：本次批量上传中已经包含内容相同的文件《{filename}》。"
                )
                create_ingest_task_file(
                    file_id=file_id,
                    task_id=task_id,
                    document_id="",
                    filename=filename,
                    file_type=extension,
                    stored_path="",
                    status="duplicate",
                    stage="duplicate",
                    error=duplicate_message,
                    duplicate_document_id=duplicate_by_hash.id if duplicate_by_hash else "",
                )
                continue

            create_ingest_task_file(
                file_id=file_id,
                task_id=task_id,
                document_id=document_id,
                filename=filename,
                file_type=extension,
                stored_path=relative_path(stored_path),
                status="queued",
                stage="uploaded",
            )
            queued_filenames.add(filename)
            queued_hashes.add(content_hash)
            queued_count += 1
        finally:
            await file.close()

    refresh_ingest_task_counters(task_id)

    if queued_count > 0:
        background_tasks.add_task(process_ingest_task, task_id)

    return build_ingest_task_response(task_id)


@app.get("/api/ingest-tasks/{task_id}", response_model=IngestTaskResponse)
def get_ingest_task(task_id: str) -> IngestTaskResponse:
    return build_ingest_task_response(task_id)


@app.get("/api/documents", response_model=DocumentListResponse)
def list_documents(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=100),
) -> DocumentListResponse:
    return build_document_list_response(page, page_size)


@app.patch(
    "/api/documents/{document_id}/tags",
    response_model=DocumentRecord,
    dependencies=[Depends(require_admin)],
)
def patch_document_tags(
    document_id: str,
    request: UpdateDocumentTagsRequest,
) -> DocumentRecord:
    find_document(document_id)
    return update_document_tags(document_id, request.tags)


@app.get("/api/vector-store/status", response_model=VectorStoreStatusResponse)
def vector_store_status() -> VectorStoreStatusResponse:
    if not chroma_available():
        return VectorStoreStatusResponse(
            provider="chroma",
            available=False,
            persist_path="data/chroma",
            collection="rag_chunks",
            chunk_count=0,
            embedding_provider=embedding_provider(),
        )

    try:
        status = get_vector_store_status()
    except Exception:
        return VectorStoreStatusResponse(
            provider="chroma",
            available=False,
            persist_path="data/chroma",
            collection="rag_chunks",
            chunk_count=0,
            embedding_provider=embedding_provider(),
        )

    return VectorStoreStatusResponse(**status, embedding_provider=embedding_provider())


@app.post(
    "/api/vector-store/rebuild",
    response_model=VectorStoreRebuildResponse,
    dependencies=[Depends(require_admin)],
)
def rebuild_vector_store() -> VectorStoreRebuildResponse:
    if not chroma_available():
        raise HTTPException(status_code=503, detail="Chroma is not available")

    reset_collection()
    documents = load_documents()
    chunk_count = 0

    for document in documents:
        chunks = load_chunks(document)
        if chunks:
            chunk_count += upsert_document_chunks(document, chunks)

    return VectorStoreRebuildResponse(
        rebuilt=True,
        provider="chroma",
        document_count=len(documents),
        chunk_count=chunk_count,
    )


@app.delete(
    "/api/documents/{document_id}",
    response_model=DeleteDocumentResponse,
    dependencies=[Depends(require_admin)],
)
def delete_document(document_id: str) -> DeleteDocumentResponse:
    documents = load_documents()
    document = next((item for item in documents if item.id == document_id), None)

    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    vector_chunks_deleted = 0
    try:
        vector_chunks_deleted = delete_document_chunks(document.id)
    except Exception:
        pass

    removed_files = [
        removed
        for removed in (
            delete_data_file(document.stored_path),
            delete_data_file(document.parsed_path),
            delete_data_file(document.chunks_path),
        )
        if removed is not None
    ]

    delete_document_record(document.id)

    return DeleteDocumentResponse(
        deleted=True,
        document_id=document.id,
        filename=document.filename,
        removed_files=removed_files,
        vector_chunks_deleted=vector_chunks_deleted,
        sqlite_deleted=True,
    )


@app.get(
    "/api/documents/{document_id}",
    response_model=DocumentDetail,
    response_model_exclude={"chunks": {"__all__": {"embedding"}}},
)
def get_document_detail(document_id: str) -> DocumentDetail:
    document = find_document(document_id)
    chunks = load_chunks(document)
    text_preview = load_parsed_preview(document)
    returned_chunks = chunks[:DETAIL_MAX_CHUNKS]

    return DocumentDetail(
        document=document,
        text_preview=text_preview,
        preview_char_count=len(text_preview),
        chunks=returned_chunks,
        returned_chunk_count=len(returned_chunks),
    )


@app.post(
    "/api/search",
    response_model=SearchResponse,
    dependencies=[Depends(rate_limit("rag_search", limit=30, window_seconds=60))],
)
def search(request: SearchRequest) -> SearchResponse:
    return search_chunks(request)


@app.post(
    "/api/chat-template-disabled",
    response_model=ChatResponse,
    include_in_schema=False,
)
def chat_template_disabled(request: ChatRequest) -> ChatResponse:
    search_response = search_chunks(SearchRequest(question=request.question, top_k=3))

    if not search_response.results:
        return ChatResponse(
            answer="当前知识库还没有可检索的文本块。请先上传并解析文档，再发送问题。",
            sources=[],
            mode="retrieval_template",
        )

    summary_lines = [
        f"{index}. 《{item.document_filename}》chunk {item.chunk_index}，相似度 {item.score:.3f}"
        for index, item in enumerate(search_response.results, start=1)
    ]
    answer = (
        "已根据你的问题完成本地向量检索。当前版本还没有接入真实 LLM，"
        "所以先返回命中的文档片段作为回答依据：\n"
        + "\n".join(summary_lines)
    )

    return ChatResponse(
        answer=answer,
        sources=[
            Source(
                title=f"{item.document_filename} / chunk {item.chunk_index}",
                content=item.content[:500],
            )
            for item in search_response.results
        ],
        mode="retrieval_template",
    )


@app.post("/api/chat-legacy-disabled", response_model=ChatResponse, include_in_schema=False)
def chat(request: ChatRequest) -> ChatResponse:
    search_response = search_chunks(
        SearchRequest(
            question=request.question,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
        )
    )

    if not search_response.results:
        return ChatResponse(
            answer="当前知识库还没有可检索的文本块。请先上传并解析文档，再发送问题。",
            sources=[],
            mode="retrieval_template",
        )

    sources = build_sources(search_response.results)
    langchain_answer = generate_rag_answer_with_langchain(
        question=request.question,
        chunks=[item.model_dump() for item in search_response.results],
    )

    if langchain_answer:
        return ChatResponse(
            answer=langchain_answer,
            sources=sources,
            mode="langchain_deepseek",
        )

    answer = call_deepseek_chat(request.question, search_response.results)

    if answer:
        return ChatResponse(
            answer=answer,
            sources=sources,
            mode="deepseek",
        )

    return ChatResponse(
        answer=build_retrieval_template_answer(request.question, search_response.results),
        sources=sources,
        mode="retrieval_template",
    )


@app.post(
    "/api/chat",
    response_model=ChatResponse,
    dependencies=[Depends(rate_limit("rag_chat", limit=12, window_seconds=60))],
)
def chat_with_strict_rag(request: ChatRequest) -> ChatResponse:
    session_id = ensure_chat_session(request.session_id, request.question)
    history = load_chat_messages(session_id)
    rewritten_question = build_contextual_question(request.question, history)
    add_chat_message(session_id, "user", request.question)

    if is_knowledge_overview_question(request.question):
        overview = build_knowledge_overview()
        response = ChatResponse(
            session_id=session_id,
            rewritten_question=request.question,
            answer=build_knowledge_overview_answer(overview),
            sources=build_knowledge_overview_sources(overview),
            mode="knowledge_overview",
            retrieval_mode="local_hash_embedding",
            score_threshold=request.score_threshold,
            workflow="manual",
            graph_path=["knowledge_overview"],
            overview=overview,
        )
        add_chat_message(session_id, "assistant", response.answer)
        response.messages = load_chat_messages(session_id)
        return response

    graph_result = run_langgraph_rag_chat(
        question=rewritten_question,
        top_k=request.top_k,
        score_threshold=request.score_threshold,
        search_fn=lambda question, top_k, score_threshold: search_chunks(
            SearchRequest(
                question=question,
                top_k=top_k,
                score_threshold=score_threshold,
            )
        ),
        build_sources_fn=build_sources,
        langchain_answer_fn=generate_rag_answer_with_langchain,
        deepseek_answer_fn=call_deepseek_chat,
        template_answer_fn=build_retrieval_template_answer,
    )

    if graph_result is not None:
        response = ChatResponse(
            session_id=session_id,
            rewritten_question=rewritten_question,
            answer=graph_result["answer"],
            sources=graph_result.get("sources", []),
            mode=graph_result["mode"],
            retrieval_mode=graph_result.get("retrieval_mode", "local_hash_embedding"),
            score_threshold=request.score_threshold,
            workflow="langgraph",
            graph_path=graph_result.get("graph_path", []),
        )
        add_chat_message(session_id, "assistant", response.answer)
        response.messages = load_chat_messages(session_id)
        return response

    search_response = search_chunks(
        SearchRequest(
            question=rewritten_question,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
        )
    )

    if not search_response.results:
        response = ChatResponse(
            session_id=session_id,
            rewritten_question=rewritten_question,
            answer=NO_ENOUGH_CONTEXT_ANSWER,
            sources=[],
            mode="retrieval_template",
            retrieval_mode=search_response.mode,
            score_threshold=request.score_threshold,
            workflow="manual",
        )
        add_chat_message(session_id, "assistant", response.answer)
        response.messages = load_chat_messages(session_id)
        return response

    sources = build_sources(search_response.results)
    langchain_answer = generate_rag_answer_with_langchain(
        question=rewritten_question,
        chunks=[item.model_dump() for item in search_response.results],
    )

    if langchain_answer:
        response = ChatResponse(
            session_id=session_id,
            rewritten_question=rewritten_question,
            answer=langchain_answer,
            sources=sources,
            mode="langchain_deepseek",
            retrieval_mode=search_response.mode,
            score_threshold=request.score_threshold,
            workflow="manual",
        )
        add_chat_message(session_id, "assistant", response.answer)
        response.messages = load_chat_messages(session_id)
        return response

    answer = call_deepseek_chat(rewritten_question, search_response.results)

    if answer:
        response = ChatResponse(
            session_id=session_id,
            rewritten_question=rewritten_question,
            answer=answer,
            sources=sources,
            mode="deepseek",
            retrieval_mode=search_response.mode,
            score_threshold=request.score_threshold,
            workflow="manual",
        )
        add_chat_message(session_id, "assistant", response.answer)
        response.messages = load_chat_messages(session_id)
        return response

    response = ChatResponse(
        session_id=session_id,
        rewritten_question=rewritten_question,
        answer=build_retrieval_template_answer(rewritten_question, search_response.results),
        sources=sources,
        mode="retrieval_template",
        retrieval_mode=search_response.mode,
        score_threshold=request.score_threshold,
        workflow="manual",
    )
    add_chat_message(session_id, "assistant", response.answer)
    response.messages = load_chat_messages(session_id)
    return response


@app.get("/api/chat/sessions/{session_id}/messages", response_model=list[ChatMessage])
def get_chat_session_messages(session_id: str) -> list[ChatMessage]:
    return load_chat_messages(normalize_session_id(session_id), limit=50)
