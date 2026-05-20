import hashlib
import json
import math
import os
import re
import shutil
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pypdf import PdfReader


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
PARSED_DIR = DATA_DIR / "parsed"
CHUNKS_DIR = DATA_DIR / "chunks"
DOCUMENTS_FILE = DATA_DIR / "documents.json"
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}

APP_VERSION = "0.8.0"
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
    char_count: int
    chunk_count: int = 0
    created_at: str


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


class SearchRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=SEARCH_TOP_K_DEFAULT, ge=1, le=SEARCH_TOP_K_MAX)


class SearchResult(BaseModel):
    document_id: str
    document_filename: str
    chunk_id: str
    chunk_index: int
    title: str = ""
    score: float
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
    total_chunks: int
    results: list[SearchResult]
    mode: Literal["local_hash_embedding"]


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=SEARCH_TOP_K_DEFAULT, ge=1, le=SEARCH_TOP_K_MAX)


class Source(BaseModel):
    title: str
    content: str
    document_id: str = ""
    chunk_id: str = ""
    score: float | None = None
    page_start: int | None = None
    page_end: int | None = None
    section_path: list[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    mode: Literal["deepseek", "retrieval_template"]


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
    if not DOCUMENTS_FILE.exists():
        DOCUMENTS_FILE.write_text("[]", encoding="utf-8")


def load_documents() -> list[DocumentRecord]:
    ensure_data_dirs()
    try:
        raw_documents = json.loads(DOCUMENTS_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raw_documents = []

    return [DocumentRecord(**item) for item in raw_documents]


def save_documents(documents: list[DocumentRecord]) -> None:
    ensure_data_dirs()
    payload = [document.model_dump() for document in documents]
    DOCUMENTS_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


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


def stable_hash_index(token: str) -> int:
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % EMBEDDING_DIM


def text_tokens(text: str) -> list[str]:
    lowered = text.lower()
    words = re.findall(r"[a-z0-9]+", lowered)
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    chinese_bigrams = [f"{a}{b}" for a, b in zip(chinese_chars, chinese_chars[1:])]
    return words + chinese_chars + chinese_bigrams


def embed_text(text: str) -> list[float]:
    vector = [0.0] * EMBEDDING_DIM
    for token in text_tokens(text):
        vector[stable_hash_index(token)] += 1.0

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector

    return [value / norm for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


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
        if not chunk.embedding:
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
                    content=chunk.content,
                    char_count=chunk.char_count,
                    strategy=chunk.strategy,
                    section_path=chunk.section_path,
                    token_estimate=chunk.token_estimate,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                )
            )

    ranked_results = sorted(results, key=lambda item: item.score, reverse=True)
    return SearchResponse(
        question=request.question,
        top_k=request.top_k,
        total_chunks=total_chunks,
        results=ranked_results[: request.top_k],
        mode="local_hash_embedding",
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


@app.post("/api/documents/upload", response_model=DocumentRecord)
async def upload_document(file: UploadFile = File(...)) -> DocumentRecord:
    ensure_data_dirs()

    original_filename = safe_filename(repair_mojibake_filename(file.filename or ""))
    extension = Path(original_filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only .txt, .md, and .pdf files are supported.",
        )

    document_id = f"doc_{uuid.uuid4().hex[:12]}"
    stored_path = UPLOAD_DIR / f"{document_id}{extension}"

    try:
        with stored_path.open("wb") as target:
            shutil.copyfileobj(file.file, target)
    finally:
        await file.close()

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
        char_count=len(parsed_text),
        chunk_count=len(chunks),
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    documents = load_documents()
    documents.append(document)
    save_documents(documents)

    return document


@app.get("/api/documents", response_model=list[DocumentRecord])
def list_documents() -> list[DocumentRecord]:
    return sorted(load_documents(), key=lambda document: document.created_at, reverse=True)


@app.delete("/api/documents/{document_id}", response_model=DeleteDocumentResponse)
def delete_document(document_id: str) -> DeleteDocumentResponse:
    documents = load_documents()
    document = next((item for item in documents if item.id == document_id), None)

    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    removed_files = [
        removed
        for removed in (
            delete_data_file(document.stored_path),
            delete_data_file(document.parsed_path),
            delete_data_file(document.chunks_path),
        )
        if removed is not None
    ]

    save_documents([item for item in documents if item.id != document_id])

    return DeleteDocumentResponse(
        deleted=True,
        document_id=document.id,
        filename=document.filename,
        removed_files=removed_files,
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


@app.post("/api/search", response_model=SearchResponse)
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


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    search_response = search_chunks(
        SearchRequest(question=request.question, top_k=request.top_k)
    )

    if not search_response.results:
        return ChatResponse(
            answer="当前知识库还没有可检索的文本块。请先上传并解析文档，再发送问题。",
            sources=[],
            mode="retrieval_template",
        )

    sources = build_sources(search_response.results)
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
