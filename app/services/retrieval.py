from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import uuid
from collections import Counter
from pathlib import Path
from threading import RLock
from typing import Any, Callable

import httpx

try:
    import jieba
except ImportError:  # pragma: no cover
    jieba = None

from app.config import ModelEndpointConfig, ROOT_DIR
from app.errors import AppError
from app.services.openai_client import OpenAICompatibleClient, chat_completions_url

try:
    import pymupdf
except ImportError:  # pragma: no cover
    pymupdf = None


def _endpoint_url(base_url: str, operation: str) -> str:
    root = base_url.strip().rstrip("/")
    if root.endswith("/chat/completions"):
        root = root[: -len("/chat/completions")]
    if root.endswith(f"/{operation}"):
        return root
    if root.endswith("/v1"):
        return f"{root}/{operation}"
    return f"{root}/v1/{operation}"


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    denominator = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return numerator / denominator if denominator else 0.0


def _pdf_fingerprint(path: Path) -> str:
    """Fingerprint of a source PDF, identical to scripts/precompute_semifinal_ocr.py."""
    stat = path.stat()
    seed = f"{path.name}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


class DocumentChunkIndex:
    """PDF text chunks cached as JSON files and loaded into process memory."""

    def __init__(self, cache_dir: Path | str | None = None, chunk_size: int = 720, overlap: int = 100) -> None:
        self.cache_dir = Path(cache_dir or ROOT_DIR / "runtime" / "rag-index")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._memory: dict[str, list[dict[str, Any]]] = {}
        self._page_memory: dict[str, list[dict[str, Any]]] = {}
        self._lock = RLock()

    def chunks(self, path: Path) -> list[dict[str, Any]]:
        resolved = path.resolve()
        stat = resolved.stat()
        fingerprint = f"{resolved}|{stat.st_size}|{stat.st_mtime_ns}|{self.chunk_size}|{self.overlap}"
        cache_key = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()
        with self._lock:
            if cache_key in self._memory:
                return [dict(item) for item in self._memory[cache_key]]
            cache_path = self.cache_dir / f"{cache_key}.json"
            cached = self._read_cache(cache_path, fingerprint)
            if cached is None:
                cached = self._extract(resolved)
                self._write_cache(cache_path, fingerprint, resolved.name, cached)
            self._memory[cache_key] = cached
            return [dict(item) for item in cached]

    def pages(self, path: Path) -> list[dict[str, Any]]:
        resolved = path.resolve()
        stat = resolved.stat()
        fingerprint = f"{resolved}|{stat.st_size}|{stat.st_mtime_ns}"
        key = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()
        with self._lock:
            if key in self._page_memory:
                return [dict(item) for item in self._page_memory[key]]
        cache_path = self.cache_dir / f"{key}-pages.json"
        if cache_path.is_file():
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                if payload.get("fingerprint") == fingerprint and isinstance(payload.get("pages"), list):
                    pages = payload["pages"]
                    with self._lock:
                        self._page_memory[key] = pages
                    return [dict(item) for item in pages]
            except (OSError, ValueError, AttributeError):
                pass
        if pymupdf is None:
            return []
        try:
            with pymupdf.open(resolved) as pdf:
                pages = [
                    {
                        "page": page_index,
                        "text": re.sub(
                            r"\n{3,}",
                            "\n\n",
                            re.sub(r"[ \t]+", " ", pdf_page.get_text("text") or ""),
                        ).strip(),
                    }
                    for page_index, pdf_page in enumerate(pdf, start=1)
                ]
        except Exception as exc:
            raise AppError("RAG_INDEX_FAILED", "PDF 文本语料读取失败", 422, True) from exc
        with self._lock:
            self._page_memory[key] = pages
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"version": 1, "fingerprint": fingerprint, "document": resolved.name, "pages": pages}, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(temporary, cache_path)
        return [dict(item) for item in pages]

    def chunk_pages(self, pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        for page in pages:
            digest_counts: Counter[str] = Counter()
            for offset, value in enumerate(self._split(str(page.get("text") or ""))):
                digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
                digest_counts[digest] += 1
                duplicate_suffix = f"-{digest_counts[digest]}" if digest_counts[digest] > 1 else ""
                chunks.append(
                    {
                        "id": f"p{page['page']}-{str(page.get('source') or 'pdf_text')}-{digest}{duplicate_suffix}",
                        "page": int(page["page"]),
                        "text": value,
                        "source": str(page.get("source") or "pdf_text"),
                    }
                )
        return chunks

    def load_ocr_cache(self, path: Path, model: str) -> dict[int, str]:
        cache_path = self._ocr_cache_path(path, model)
        if not cache_path.is_file():
            return {}
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            return {int(page): str(text) for page, text in payload.get("pages", {}).items() if str(text).strip()}
        except (OSError, ValueError, TypeError, AttributeError):
            return {}

    def save_ocr_cache(self, path: Path, model: str, pages: dict[int, str]) -> None:
        cache_path = self._ocr_cache_path(path, model)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"version": 1, "document": path.name, "model": model, "pages": pages}, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(temporary, cache_path)

    @staticmethod
    def load_txt_ocr_corpus(
        path: Path,
        expected_model: str | None = None,
        *,
        allow_model_mismatch: bool = False,
    ) -> dict[str, Any]:
        """Load the precomputed OCR corpus of a PDF together with its identity.

        The corpus under runtime/ocr-corpus is a bulk cache written by
        scripts/precompute_semifinal_ocr.py. It may only be reused as OCR
        evidence when the manifest still matches the current PDF fingerprint
        and the OCR model that is actually configured, so a page is never
        attributed to a model that did not produce it.
        """
        resolved = path.resolve()
        corpus_root = ROOT_DIR / "runtime" / "ocr-corpus"
        split = resolved.parent.parent.name if resolved.parent.name == "documents" else "shared"
        preferred = corpus_root / split / resolved.stem
        # Cache identity is the PDF filename stem. Prefer the dataset split
        # implied by the source path, then fall back to any train/test cache
        # with the same stem so preliminary and semi-final OCR can share the
        # precomputed TXT corpus.
        directories = [preferred]
        if not preferred.is_dir() and corpus_root.is_dir():
            directories.extend(
                item
                for item in sorted(corpus_root.glob("*/" + resolved.stem))
                if item != preferred
            )
        directory = next((item for item in directories if item.is_dir()), None)
        meta: dict[str, Any] = {
            "source": "none",
            "directory": None,
            "document": resolved.name,
            "model": None,
            "expected_model": expected_model,
            "model_match": None,
            "fingerprint_match": None,
            "page_count": None,
            "completed_pages": 0,
            "stale": False,
            "stale_reason": None,
        }
        if directory is None:
            return {"pages": {}, "meta": meta}

        manifest = _read_json_object(directory / "manifest.json")
        if manifest is None:
            meta.update(
                {
                    "source": "precomputed_corpus",
                    "directory": str(directory),
                    "stale": True,
                    "stale_reason": "语料缺少 manifest，无法校验来源",
                }
            )
            return {"pages": {}, "meta": meta}

        corpus_model = str((manifest.get("ocr") or {}).get("model") or "") or None
        fingerprint = str((manifest.get("source") or {}).get("fingerprint") or "")
        current_fingerprint = _pdf_fingerprint(resolved)
        model_match = corpus_model == expected_model if expected_model else None
        fingerprint_match = bool(fingerprint) and fingerprint == current_fingerprint
        meta.update(
            {
                "source": "precomputed_corpus",
                "directory": str(directory),
                "document": str(manifest.get("document") or resolved.name),
                "model": corpus_model,
                "model_match": model_match,
                "fingerprint_match": fingerprint_match,
                "page_count": manifest.get("page_count"),
                "completed_pages": int(manifest.get("completed_pages") or 0),
            }
        )
        reasons: list[str] = []
        if model_match is False:
            reasons.append(f"OCR 模型已从 {corpus_model or '未知'} 切换为 {expected_model}")
        if not fingerprint_match:
            reasons.append("源 PDF 指纹与语料记录不一致")
        if reasons:
            meta.update({"stale": True, "stale_reason": "；".join(reasons)})
            if not allow_model_mismatch:
                return {"pages": {}, "meta": meta}

        pages: dict[int, str] = {}
        page_states = manifest.get("pages") if isinstance(manifest.get("pages"), dict) else {}
        for item in sorted(directory.glob("page-*.txt")):
            match = re.fullmatch(r"page-(\d+)\.txt", item.name, flags=re.IGNORECASE)
            if not match:
                continue
            page_number = int(match.group(1))
            state = page_states.get(str(page_number))
            if isinstance(state, dict) and state.get("status") != "completed":
                continue
            try:
                text = item.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if text:
                pages[page_number] = text
        meta["completed_pages"] = len(pages)
        return {"pages": pages, "meta": meta}

    def _ocr_cache_path(self, path: Path, model: str) -> Path:
        resolved = path.resolve()
        stat = resolved.stat()
        value = f"{resolved}|{stat.st_size}|{stat.st_mtime_ns}|ocr|{model}"
        return self.cache_dir / f"{hashlib.sha256(value.encode('utf-8')).hexdigest()}-ocr.json"

    def _extract(self, path: Path) -> list[dict[str, Any]]:
        if pymupdf is None:
            return []
        chunks: list[dict[str, Any]] = []
        try:
            with pymupdf.open(path) as pdf:
                for page_index, pdf_page in enumerate(pdf, start=1):
                    text = re.sub(r"[ \t]+", " ", pdf_page.get_text("text") or "")
                    text = re.sub(r"\n{3,}", "\n\n", text).strip()
                    for offset, value in enumerate(self._split(text)):
                        chunks.append(
                            {
                                "id": f"p{page_index}-c{offset + 1}",
                                "page": page_index,
                                "text": value,
                                "source": "pdf_text",
                            }
                        )
        except Exception as exc:
            raise AppError("RAG_INDEX_FAILED", "PDF 文本索引构建失败", 422, True) from exc
        return chunks

    def _split(self, text: str) -> list[str]:
        if not text:
            return []
        normalized = re.sub(r"[ \t]+", " ", text)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.overlap,
                length_function=len,
                keep_separator="end",
                separators=[
                    "\n\n",
                    "\n",
                    "。",
                    "；",
                    "！",
                    "？",
                    ";",
                    ". ",
                    "，",
                    ",",
                    " ",
                    "",
                ],
            )
            headings = list(
                re.finditer(
                    r"(?m)^(?=摘要(?:\s|$)|技术领域(?:\s|$)|背景技术(?:\s|$)|发明内容(?:\s|$)|附图说明(?:\s|$)|具体实施方式(?:\s|$)|权利要求(?:书)?\s*\d*)",
                    normalized,
                )
            )
            sections: list[str] = []
            if headings:
                if headings[0].start() > 0:
                    sections.append(normalized[: headings[0].start()])
                for index, match in enumerate(headings):
                    end = headings[index + 1].start() if index + 1 < len(headings) else len(normalized)
                    sections.append(normalized[match.start() : end])
            else:
                sections = [normalized]
            return [
                value.strip()
                for section in sections
                for value in splitter.split_text(section.strip())
                if value.strip()
            ]
        except ImportError:
            pass
        values: list[str] = []
        start = 0
        while start < len(normalized):
            end = min(len(normalized), start + self.chunk_size)
            if end < len(normalized):
                boundary = max(
                    normalized.rfind("\n", start, end),
                    normalized.rfind("。", start, end),
                    normalized.rfind("；", start, end),
                )
                if boundary > start + self.chunk_size // 2:
                    end = boundary + 1
            value = normalized[start:end].strip()
            if value:
                values.append(value)
            if end >= len(text):
                break
            start = max(start + 1, end - self.overlap)
        return values

    @staticmethod
    def _read_cache(path: Path, fingerprint: str) -> list[dict[str, Any]] | None:
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("fingerprint") == fingerprint and isinstance(payload.get("chunks"), list):
                return payload["chunks"]
        except (OSError, ValueError, AttributeError):
            return None
        return None

    def _write_cache(self, path: Path, fingerprint: str, document: str, chunks: list[dict[str, Any]]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        payload = {"version": 1, "fingerprint": fingerprint, "document": document, "chunks": chunks}
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, path)


class LangChainFaissIndex:
    """LangChain embeddings with safe NPY persistence and an in-memory FAISS index."""

    def __init__(self, cache_dir: Path | str | None = None) -> None:
        self.cache_dir = Path(cache_dir or ROOT_DIR / "runtime" / "rag-index" / "faiss")
        self._locks: dict[str, RLock] = {}
        self._locks_guard = RLock()

    @staticmethod
    def _components():
        try:
            from langchain_core.documents import Document
            from langchain_openai import OpenAIEmbeddings
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            import faiss
            import numpy as np
        except ImportError as exc:
            raise AppError(
                "RAG_VECTOR_DEPENDENCY_MISSING",
                "缺少 LangChain 或 FAISS 依赖，向量索引已明确降级",
                503,
                False,
                {"dependency": str(exc)},
            ) from exc
        return Document, RecursiveCharacterTextSplitter, OpenAIEmbeddings, faiss, np

    async def search(
        self,
        question: str,
        chunks: list[dict[str, Any]],
        document_path: Path,
        config: ModelEndpointConfig,
        *,
        ocr_model: str,
        chunk_size: int,
        overlap: int,
        limit: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return await asyncio.to_thread(
            self._search_sync,
            question,
            chunks,
            document_path,
            config,
            ocr_model,
            chunk_size,
            overlap,
            limit,
        )

    def _search_sync(
        self,
        question: str,
        chunks: list[dict[str, Any]],
        document_path: Path,
        config: ModelEndpointConfig,
        ocr_model: str,
        chunk_size: int,
        overlap: int,
        limit: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        Document, RecursiveCharacterTextSplitter, OpenAIEmbeddings, faiss, np = self._components()
        key, manifest = self._manifest(
            chunks, document_path, config, ocr_model=ocr_model, chunk_size=chunk_size, overlap=overlap
        )
        target = self.cache_dir / key
        lock = self._lock_for(key)
        try:
            embeddings = OpenAIEmbeddings(
                model=config.model,
                api_key=config.api_key,
                base_url=self._openai_root(config.base_url),
                timeout=config.timeout,
                check_embedding_ctx_length=False,
                chunk_size=32,
                max_retries=2,
            )
            with lock:
                cache_hit = self._valid_index(target, manifest, np)
                if cache_hit:
                    matrix = np.load(target / "vectors.npy", allow_pickle=False)
                    documents = json.loads((target / "documents.json").read_text(encoding="utf-8"))
                else:
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=overlap,
                        length_function=len,
                        keep_separator="end",
                        separators=["\n\n", "\n", "。", "；", "！", "？", "，", " ", ""],
                    )
                    documents: list[dict[str, Any]] = []
                    for item in chunks:
                        pieces = splitter.split_text(str(item["text"]))
                        for index, text in enumerate(pieces, start=1):
                            document = dict(item)
                            if len(pieces) > 1:
                                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
                                document["id"] = f"{item['id']}-lc{index}-{digest}"
                            document["text"] = text
                            documents.append(document)
                    raw_vectors = embeddings.embed_documents([item["text"] for item in documents])
                    matrix = np.asarray(raw_vectors, dtype="float32")
                    if matrix.ndim != 2 or matrix.shape[0] != len(documents) or matrix.shape[1] < 1:
                        raise AppError("MODEL_RESPONSE_INVALID", "Embedding 向量维度不合法", 502, True)
                    faiss.normalize_L2(matrix)
                    manifest = {
                        **manifest,
                        "vector_dimension": int(matrix.shape[1]),
                        "vector_count": int(matrix.shape[0]),
                        "documents_sha256": self._documents_digest(documents),
                    }
                    self._atomic_save(matrix, documents, target, manifest, np)
                vector_index = faiss.IndexFlatIP(int(matrix.shape[1]))
                vector_index.add(matrix)
                query_vector = np.asarray([embeddings.embed_query(question)], dtype="float32")
                if query_vector.shape != (1, vector_index.d):
                    raise AppError("MODEL_RESPONSE_INVALID", "查询向量维度与 FAISS 索引不一致", 502, True)
                faiss.normalize_L2(query_vector)
                scores, indices = vector_index.search(
                    query_vector, min(limit, len(documents))
                )
        except AppError:
            raise
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            response = getattr(exc, "response", None)
            status_code = status_code or getattr(response, "status_code", None)
            if status_code in (401, 403):
                raise AppError("MODEL_AUTH_FAILED", "Retrieval embedding 模型鉴权失败", 502, False) from exc
            if status_code == 429:
                raise AppError("MODEL_RATE_LIMITED", "Retrieval embedding 模型服务限流", 503, True) from exc
            raise AppError(
                "RAG_VECTOR_INDEX_UNAVAILABLE",
                "LangChain FAISS 向量索引暂不可用，已允许明确降级",
                502,
                True,
                {"exception": type(exc).__name__},
            ) from exc
        items: list[dict[str, Any]] = []
        for score, index in zip(scores[0].tolist(), indices[0].tolist()):
            if index < 0 or index >= len(documents):
                continue
            item = dict(documents[index])
            item["vector_score"] = round(float(score), 6)
            items.append(item)
        return items, {
            "vector_index_status": "ready",
            "vector_index_cache": "hit" if cache_hit else "built",
            "vector_index_key": key,
            "vector_persistence": "vectors.npy",
            "degradation_reason": None,
        }

    def _manifest(
        self,
        chunks: list[dict[str, Any]],
        document_path: Path,
        config: ModelEndpointConfig,
        *,
        ocr_model: str,
        chunk_size: int,
        overlap: int,
    ) -> tuple[str, dict[str, Any]]:
        resolved = document_path.resolve()
        stat = resolved.stat()
        chunks_digest = hashlib.sha256(
            json.dumps(chunks, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        manifest = {
            "version": 1,
            "document": resolved.name,
            "document_size": stat.st_size,
            "document_mtime_ns": stat.st_mtime_ns,
            "chunks_digest": chunks_digest,
            "chunk_count": len(chunks),
            "chunk_size": chunk_size,
            "overlap": overlap,
            "ocr_model": ocr_model,
            "embedding_model": config.model,
            "embedding_base_url_hash": hashlib.sha256(
                self._openai_root(config.base_url).encode("utf-8")
            ).hexdigest()[:16],
        }
        key = hashlib.sha256(
            json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        return key, manifest

    @staticmethod
    def _openai_root(base_url: str) -> str:
        root = base_url.strip().rstrip("/")
        for suffix in ("/chat/completions", "/embeddings"):
            if root.endswith(suffix):
                root = root[: -len(suffix)]
                break
        return root if root.endswith("/v1") else f"{root}/v1"

    def _lock_for(self, key: str) -> RLock:
        with self._locks_guard:
            return self._locks.setdefault(key, RLock())

    @classmethod
    def _valid_index(cls, target: Path, expected: dict[str, Any], np: Any) -> bool:
        manifest_path = target / "manifest.json"
        documents_path = target / "documents.json"
        vectors_path = target / "vectors.npy"
        if not (manifest_path.is_file() and documents_path.is_file() and vectors_path.is_file()):
            return False
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for key, value in expected.items():
                if manifest.get(key) != value:
                    return False
            documents = json.loads(documents_path.read_text(encoding="utf-8"))
            if not isinstance(documents, list) or cls._documents_digest(documents) != manifest.get("documents_sha256"):
                return False
            matrix = np.load(vectors_path, allow_pickle=False)
            return (
                matrix.ndim == 2
                and int(matrix.shape[0]) == len(documents) == int(manifest.get("vector_count", -1))
                and int(matrix.shape[1]) == int(manifest.get("vector_dimension", -1))
                and str(matrix.dtype) == "float32"
            )
        except (OSError, ValueError, TypeError, AttributeError, RuntimeError):
            return False

    @staticmethod
    def _documents_digest(documents: list[dict[str, Any]]) -> str:
        return hashlib.sha256(
            json.dumps(documents, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def _atomic_save(
        self,
        matrix: Any,
        documents: list[dict[str, Any]],
        target: Path,
        manifest: dict[str, Any],
        np: Any,
    ) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.cache_dir / f".{target.name}.tmp-{uuid.uuid4().hex}"
        stale: Path | None = None
        try:
            temporary.mkdir(parents=False, exist_ok=False)
            np.save(temporary / "vectors.npy", matrix, allow_pickle=False)
            (temporary / "documents.json").write_text(
                json.dumps(documents, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            (temporary / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            try:
                os.replace(temporary, target)
            except OSError:
                # Another process may have published the same immutable,
                # content-addressed index while this process was building it.
                if self._valid_index(target, manifest, np):
                    return
                if target.exists():
                    stale = self.cache_dir / f".{target.name}.stale-{uuid.uuid4().hex}"
                    os.replace(target, stale)
                os.replace(temporary, target)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary, ignore_errors=True)
            if stale is not None and stale.exists():
                shutil.rmtree(stale, ignore_errors=True)


class RagRetrievalService:
    def __init__(
        self,
        index: DocumentChunkIndex | None = None,
        vector_index: LangChainFaissIndex | None = None,
    ) -> None:
        self.index = index or DocumentChunkIndex()
        base_cache = Path(getattr(self.index, "cache_dir", ROOT_DIR / "runtime" / "rag-index"))
        self.vector_index = vector_index or LangChainFaissIndex(base_cache / "faiss")

    @staticmethod
    def capabilities() -> dict[str, Any]:
        dependencies = {
            "langchain_core": importlib.util.find_spec("langchain_core") is not None,
            "langchain_text_splitters": importlib.util.find_spec("langchain_text_splitters") is not None,
            "langchain_openai": importlib.util.find_spec("langchain_openai") is not None,
            "faiss": importlib.util.find_spec("faiss") is not None,
        }
        return {
            "framework": "LangChain",
            "document_model": "langchain_core.documents.Document",
            "text_splitter": "RecursiveCharacterTextSplitter",
            "vector_store": "FAISS IndexFlatIP (runtime)",
            "vector_index": "IndexFlatIP",
            "persistence": "vectors.npy + documents.json + manifest.json",
            "retrieval_strategy": "BM25-style keyword + embedding vector score fusion",
            "database_required": False,
            "available": all(dependencies.values()),
            "langchain_available": all(
                dependencies[name]
                for name in ("langchain_core", "langchain_text_splitters", "langchain_openai")
            ),
            "faiss_available": dependencies["faiss"],
            "dependencies": dependencies,
        }

    async def search(
        self,
        question: str,
        path: Path,
        retrieval_config: ModelEndpointConfig,
        reranker_config: ModelEndpointConfig,
        ocr_config: ModelEndpointConfig,
        *,
        target_page: int,
        target_ocr: str = "",
        render_page: Callable[[int], bytes] | None = None,
        top_k: int = 6,
    ) -> dict[str, Any]:
        chunks, corpus = await self._hybrid_corpus(
            path,
            ocr_config,
            target_page=target_page,
            target_ocr=target_ocr,
            render_page=render_page,
        )
        if not chunks:
            return {
                "items": [],
                "retrieval_protocol": "empty_index",
                "rerank_protocol": "skipped",
                "chunk_count": 0,
                "recalled_count": 0,
                "reranked_count": 0,
                **corpus,
            }
        keyword_candidates = self._keyword_rank(question, chunks, limit=min(24, len(chunks)))
        try:
            vector_candidates, vector_status = await self.vector_index.search(
                question,
                chunks,
                path,
                retrieval_config,
                ocr_model=ocr_config.model,
                chunk_size=self.index.chunk_size,
                overlap=self.index.overlap,
                limit=min(12, len(chunks)),
            )
            recalled = self._fuse_candidates(
                keyword_candidates,
                vector_candidates,
                limit=min(12, len(chunks)),
            )
            retrieval_protocol = "hybrid_keyword_vector_npy"
        except AppError as exc:
            recalled = [dict(item) for item in keyword_candidates[: min(12, len(keyword_candidates))]]
            for item in recalled:
                item["retrieval_score"] = item.get("keyword_score", 0.0)
            retrieval_protocol = "keyword_only_degraded"
            vector_status = {
                "vector_index_status": "degraded",
                "vector_index_cache": None,
                "vector_index_key": None,
                "vector_persistence": "vectors.npy",
                "degradation_reason": {"code": exc.code, "message": exc.message},
            }
        try:
            rerank_scores = await self._native_rerank(question, recalled, reranker_config)
            rerank_protocol = "compatible_rerank"
            rerank_degradation_reason = None
        except AppError as exc:
            rerank_scores = [float(item.get("retrieval_score", 0.0)) for item in recalled]
            rerank_protocol = "retrieval_score_fallback"
            rerank_degradation_reason = {"code": exc.code, "message": exc.message}
        ranked = self._rank(recalled, rerank_scores, limit=min(top_k, len(recalled)), score_key="rerank_score")
        degradation_reason = vector_status.get("degradation_reason")
        degraded = bool(degradation_reason) or rerank_degradation_reason is not None
        return {
            "items": ranked,
            "retrieval_protocol": retrieval_protocol,
            "rerank_protocol": rerank_protocol,
            "rerank_degradation_reason": rerank_degradation_reason,
            "chunk_count": len(chunks),
            "recalled_count": len(recalled),
            "reranked_count": len(ranked),
            # Single explicit degradation flag plus the index version actually
            # used, so clients never have to infer the state from protocols.
            "degraded": degraded,
            "degradation_reason": degradation_reason,
            "index_version": vector_status.get("vector_index_key"),
            **vector_status,
            **corpus,
        }

    async def test_connection(self, kind: str, config: ModelEndpointConfig) -> dict[str, Any]:
        if not config.configured:
            raise AppError("MODEL_NOT_CONFIGURED", f"{kind.upper()} 模型配置不完整", 409)
        if kind == "retrieval":
            vectors = await self._embeddings(["工业文档检索连通测试"], config)
            return {
                "service": kind,
                "status": "connected",
                "model": config.model,
                "protocol": "openai_embeddings",
                "dimensions": len(vectors[0]),
            }
        scores = await self._native_rerank(
            "工业文档检索",
            [{"text": "工业设备维护说明"}, {"text": "无关生活信息"}],
            config,
        )
        return {
            "service": kind,
            "status": "connected",
            "model": config.model,
            "protocol": "compatible_rerank",
            "sample_scores": scores,
        }

    async def _hybrid_corpus(
        self,
        path: Path,
        ocr_config: ModelEndpointConfig,
        *,
        target_page: int,
        target_ocr: str,
        render_page: Callable[[int], bytes] | None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        native_pages = self.index.pages(path)
        json_ocr_cache = self.index.load_ocr_cache(path, ocr_config.model)
        load_precomputed = getattr(self.index, "load_txt_ocr_corpus", None)
        corpus: dict[str, Any] = {"pages": {}, "meta": {"source": "none", "stale": False, "stale_reason": None}}
        if callable(load_precomputed):
            corpus = load_precomputed(path, ocr_config.model)
        precomputed_ocr_cache = corpus.get("pages") or {}
        ocr_cache = {**json_ocr_cache, **precomputed_ocr_cache}
        original_json_ocr_cache = dict(json_ocr_cache)
        ocr_warnings: list[dict[str, Any]] = []
        target_text = re.sub(r"^===== PDF P\.\d+ =====\s*", "", target_ocr.strip())
        if target_text:
            ocr_cache[target_page] = target_text
            json_ocr_cache[target_page] = target_text
        short_pages = [page for page in native_pages if len(page["text"].strip()) < 80]
        missing = [page["page"] for page in short_pages if page["page"] not in ocr_cache]
        if missing and render_page is not None and ocr_config.configured:
            ocr_client = OpenAICompatibleClient(ocr_config, "ocr")
            for page_number in missing:
                try:
                    result = await ocr_client.chat(
                        f"请识别这份工业技术文档的第 {page_number} 页，用于构建可检索语料。保留部件编号、名称、功能描述和图中标注。",
                        "你是工业文档 OCR 引擎，只输出可见文本；无法辨认处标记为[无法辨认]。",
                        render_page(page_number),
                        max_tokens=2400,
                    )
                except AppError as exc:
                    ocr_warnings.append({"page": page_number, "code": exc.code, "message": exc.message})
                    continue
                if result["content"].strip():
                    value = result["content"].strip()
                    ocr_cache[page_number] = value
                    json_ocr_cache[page_number] = value
        if json_ocr_cache != original_json_ocr_cache:
            self.index.save_ocr_cache(path, ocr_config.model, json_ocr_cache)
        corpus_pages: list[dict[str, Any]] = []
        pdf_text_pages: list[int] = []
        ocr_pages: list[int] = []
        for page in native_pages:
            page_number = int(page["page"])
            native_text = page["text"].strip()
            if len(native_text) >= 80:
                corpus_pages.append({"page": page_number, "text": native_text, "source": "pdf_text"})
                pdf_text_pages.append(page_number)
            elif ocr_cache.get(page_number, "").strip():
                corpus_pages.append({"page": page_number, "text": ocr_cache[page_number].strip(), "source": "ocr"})
                ocr_pages.append(page_number)
            elif native_text:
                corpus_pages.append({"page": page_number, "text": native_text, "source": "pdf_text"})
                pdf_text_pages.append(page_number)
        sources = [source for source, pages in (("pdf_text", pdf_text_pages), ("ocr", ocr_pages)) if pages]
        metadata = {
            "corpus_sources": sources,
            "pdf_text_pages": pdf_text_pages,
            "ocr_pages": ocr_pages,
            "precomputed_ocr_pages": sorted(precomputed_ocr_cache),
            "ocr_warnings": ocr_warnings,
            "precomputed_ocr_corpus": corpus.get("meta"),
        }
        return self.index.chunk_pages(corpus_pages), metadata

    @staticmethod
    def _lexical_prefilter(question: str, chunks: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        return RagRetrievalService._keyword_rank(question, chunks, limit)

    @staticmethod
    def _tokens(text: str) -> list[str]:
        normalized = str(text or "").lower()
        words = list(jieba.cut(normalized, cut_all=False)) if jieba is not None else re.findall(
            r"[a-z0-9]+|[\u4e00-\u9fff]+", normalized
        )
        tokens = [word.strip() for word in words if re.search(r"[a-z0-9\u4e00-\u9fff]", word)]
        chinese = "".join(re.findall(r"[\u4e00-\u9fff]", normalized))
        tokens.extend(chinese[index : index + 2] for index in range(max(0, len(chinese) - 1)))
        return tokens

    @classmethod
    def _keyword_rank(cls, question: str, chunks: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        query_tokens = cls._tokens(question)
        if not query_tokens:
            return [dict(item, keyword_score=0.0) for item in chunks[:limit]]
        tokenized = [cls._tokens(item.get("text", "")) for item in chunks]
        document_frequency = Counter(token for values in tokenized for token in set(values))
        average_length = sum(len(values) for values in tokenized) / max(1, len(tokenized))
        query_counts = Counter(query_tokens)
        scored: list[dict[str, Any]] = []
        for chunk, values in zip(chunks, tokenized):
            counts = Counter(values)
            length = max(1, len(values))
            score = 0.0
            for token, query_weight in query_counts.items():
                frequency = counts.get(token, 0)
                if not frequency:
                    continue
                inverse_frequency = math.log(1 + (len(chunks) - document_frequency[token] + 0.5) / (document_frequency[token] + 0.5))
                denominator = frequency + 1.5 * (1 - 0.75 + 0.75 * length / max(1.0, average_length))
                score += query_weight * inverse_frequency * frequency * 2.5 / denominator
            item = dict(chunk)
            item["keyword_score"] = round(score, 6)
            scored.append(item)
        scored.sort(key=lambda item: (item["keyword_score"], -int(item.get("page", 0))), reverse=True)
        return scored[:limit]

    @staticmethod
    def _fuse_candidates(
        keyword_candidates: list[dict[str, Any]],
        vector_candidates: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        keyword_max = max((float(item.get("keyword_score", 0.0)) for item in keyword_candidates), default=0.0)
        by_id: dict[str, dict[str, Any]] = {}
        keyword_rank = {str(item.get("id")): rank for rank, item in enumerate(keyword_candidates, start=1)}
        vector_rank = {str(item.get("id")): rank for rank, item in enumerate(vector_candidates, start=1)}
        for item in [*keyword_candidates, *vector_candidates]:
            key = str(item.get("id"))
            by_id.setdefault(key, {}).update(item)
        fused: list[dict[str, Any]] = []
        for key, item in by_id.items():
            keyword_score = float(item.get("keyword_score", 0.0))
            vector_score = float(item.get("vector_score", 0.0))
            item["keyword_score"] = round(keyword_score, 6)
            item["vector_score"] = round(vector_score, 6)
            keyword_normalized = keyword_score / keyword_max if keyword_max > 0 else 0.0
            vector_normalized = max(0.0, min(1.0, (vector_score + 1.0) / 2.0))
            reciprocal_rank = (
                (1 / (60 + keyword_rank[key]) if key in keyword_rank else 0.0)
                + (1 / (60 + vector_rank[key]) if key in vector_rank else 0.0)
            )
            item["retrieval_score"] = round(
                0.42 * keyword_normalized + 0.48 * vector_normalized + 3.0 * reciprocal_rank,
                6,
            )
            fused.append(item)
        fused.sort(key=lambda item: item["retrieval_score"], reverse=True)
        return fused[:limit]

    async def _retrieve_scores(
        self, question: str, chunks: list[dict[str, Any]], config: ModelEndpointConfig
    ) -> tuple[list[float], str]:
        try:
            embeddings = await self._embeddings([question, *(item["text"] for item in chunks)], config)
            return [_cosine(embeddings[0], vector) for vector in embeddings[1:]], "openai_embeddings"
        except AppError:
            ranked = self._keyword_rank(question, chunks, len(chunks))
            scores_by_id = {str(item.get("id", index)): float(item.get("keyword_score", 0.0)) for index, item in enumerate(ranked)}
            return [scores_by_id.get(str(item.get("id", index)), 0.0) for index, item in enumerate(chunks)], "keyword_only_degraded"

    async def _rerank_scores(
        self, question: str, chunks: list[dict[str, Any]], config: ModelEndpointConfig
    ) -> tuple[list[float], str]:
        try:
            return await self._native_rerank(question, chunks, config), "compatible_rerank"
        except AppError:
            return [float(item.get("retrieval_score", 0.0)) for item in chunks], "retrieval_score_fallback"

    async def _embeddings(self, inputs: list[str], config: ModelEndpointConfig) -> list[list[float]]:
        payload = await self._post_json(
            _endpoint_url(config.base_url, "embeddings"),
            config,
            {"model": config.model, "input": inputs},
        )
        try:
            rows = sorted(payload["data"], key=lambda item: item.get("index", 0))
            vectors = [[float(value) for value in item["embedding"]] for item in rows]
        except (KeyError, TypeError, ValueError) as exc:
            raise AppError("MODEL_RESPONSE_INVALID", "Embedding 响应格式不合法", 502, True) from exc
        if len(vectors) != len(inputs):
            raise AppError("MODEL_RESPONSE_INVALID", "Embedding 返回数量与输入不一致", 502, True)
        return vectors

    async def _native_rerank(
        self, question: str, chunks: list[dict[str, Any]], config: ModelEndpointConfig
    ) -> list[float]:
        payload = await self._post_json(
            _endpoint_url(config.base_url, "rerank"),
            config,
            {
                "model": config.model,
                "query": question,
                "documents": [item["text"] for item in chunks],
                "top_n": len(chunks),
                "return_documents": False,
            },
        )
        scores = [0.0] * len(chunks)
        try:
            rows = payload.get("results") or payload.get("data")
            for row in rows:
                scores[int(row["index"])] = float(row.get("relevance_score", row.get("score")))
        except (TypeError, KeyError, ValueError, IndexError) as exc:
            raise AppError("MODEL_RESPONSE_INVALID", "Rerank 响应格式不合法", 502, True) from exc
        return scores

    async def _chat_scores(
        self, question: str, chunks: list[dict[str, Any]], config: ModelEndpointConfig, service: str
    ) -> list[float]:
        rows = "\n".join(f"[{index}] {item['text'][:1200]}" for index, item in enumerate(chunks))
        prompt = (
            f"查询：{question}\n候选片段：\n{rows}\n"
            f"请对每个候选与查询的相关性评分，严格返回 JSON：{{\"scores\":[{{\"index\":0,\"score\":0.0}}]}}。"
            "score 范围 0 到 1，必须覆盖全部 index。"
        )
        result = await OpenAICompatibleClient(config, service).chat(
            prompt,
            "你是工业文档检索评分器，只返回结构化相关性分数，不回答用户问题。",
            max_tokens=max(400, len(chunks) * 40),
            json_mode=True,
        )
        try:
            payload = json.loads(result["content"])
            scores = [0.0] * len(chunks)
            for row in payload["scores"]:
                scores[int(row["index"])] = max(0.0, min(1.0, float(row["score"])))
            return scores
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            raise AppError("MODEL_RESPONSE_INVALID", f"{service} Chat 评分响应不合法", 502, True) from exc

    @staticmethod
    def _rank(
        chunks: list[dict[str, Any]], scores: list[float], limit: int, score_key: str
    ) -> list[dict[str, Any]]:
        values = []
        for index, chunk in enumerate(chunks):
            item = dict(chunk)
            item[score_key] = round(scores[index] if index < len(scores) else 0.0, 6)
            values.append(item)
        values.sort(key=lambda item: item[score_key], reverse=True)
        return values[:limit]

    @staticmethod
    async def _post_json(url: str, config: ModelEndpointConfig, body: dict[str, Any]) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=config.timeout) as client:
                response = await client.post(
                    url,
                    headers={"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"},
                    json=body,
                )
        except httpx.TimeoutException as exc:
            raise AppError("MODEL_TIMEOUT", "检索模型请求超时", 504, True) from exc
        except httpx.RequestError as exc:
            raise AppError("MODEL_UNREACHABLE", "无法连接检索模型服务", 502, True) from exc
        if response.status_code in (401, 403):
            raise AppError("MODEL_AUTH_FAILED", "检索模型鉴权失败", 502, False)
        if response.status_code == 429:
            raise AppError("MODEL_RATE_LIMITED", "检索模型服务限流", 503, True)
        if response.status_code >= 500:
            raise AppError("MODEL_UPSTREAM_ERROR", "检索模型服务异常", 502, True)
        if response.status_code >= 400:
            raise AppError("MODEL_REQUEST_REJECTED", "检索模型不支持该专用协议", 502, False)
        try:
            return response.json()
        except ValueError as exc:
            raise AppError("MODEL_RESPONSE_INVALID", "检索模型返回非 JSON 响应", 502, True) from exc
