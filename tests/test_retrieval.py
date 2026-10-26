import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from app.config import ModelEndpointConfig
from app.errors import AppError
from app.services import retrieval as retrieval_module
from app.services.retrieval import (
    DocumentChunkIndex,
    LangChainFaissIndex,
    RagRetrievalService,
    _endpoint_url,
)


def _config(model: str) -> ModelEndpointConfig:
    return ModelEndpointConfig(
        base_url="https://models.example/v1",
        api_key="secret",
        model=model,
        timeout=30,
    )


class _IndexStub:
    def __init__(self) -> None:
        self.cached = {}
        self.saved = None
        self.cache_dir = Path("runtime/test-rag-cache")
        self.chunk_size = 900
        self.overlap = 120

    def pages(self, path):
        del path
        return [
            {"page": 1, "text": "工业装置功能说明" * 12},
            {"page": 2, "text": "短页"},
            {"page": 3, "text": ""},
        ]

    def load_ocr_cache(self, path, model):
        del path, model
        return dict(self.cached)

    def save_ocr_cache(self, path, model, pages):
        del path, model
        self.saved = dict(pages)
        self.cached = dict(pages)

    def chunk_pages(self, pages):
        return [
            {"id": f"p{page['page']}-c1", "page": page["page"], "text": page["text"], "source": page["source"]}
            for page in pages
        ]


def test_hybrid_corpus_prefers_native_text_and_caches_ocr_for_short_pages(monkeypatch) -> None:
    calls = []

    async def fake_chat(self, prompt, system_prompt, image_png=None, max_tokens=1600, json_mode=False):
        del system_prompt, max_tokens, json_mode
        calls.append((prompt, image_png))
        page = image_png.decode().split("-")[-1]
        return {"content": f"OCR page {page} 部件说明", "model": self.config.model, "usage": None, "duration_ms": 1.0}

    monkeypatch.setattr(retrieval_module.OpenAICompatibleClient, "chat", fake_chat)
    index = _IndexStub()
    service = RagRetrievalService(index)
    chunks, metadata = asyncio.run(
        service._hybrid_corpus(
            Path("sample.pdf"),
            _config("ocr-model"),
            target_page=2,
            target_ocr="目标页 OCR 已存在",
            render_page=lambda page: f"image-{page}".encode(),
        )
    )

    assert calls == [(calls[0][0], b"image-3")]
    assert metadata["corpus_sources"] == ["pdf_text", "ocr"]
    assert metadata["pdf_text_pages"] == [1]
    assert metadata["ocr_pages"] == [2, 3]
    assert metadata["precomputed_ocr_pages"] == []
    assert metadata["ocr_warnings"] == []
    assert [item["source"] for item in chunks] == ["pdf_text", "ocr", "ocr"]
    assert index.saved == {2: "目标页 OCR 已存在", 3: "OCR page 3 部件说明"}

    calls.clear()
    asyncio.run(
        service._hybrid_corpus(
            Path("sample.pdf"),
            _config("ocr-model"),
            target_page=2,
            target_ocr="",
            render_page=lambda page: f"image-{page}".encode(),
        )
    )
    assert calls == []


def test_retrieval_uses_embeddings_and_reranker_prefers_native_protocol(monkeypatch) -> None:
    service = RagRetrievalService(_IndexStub())

    async def fake_embeddings(inputs, config):
        del config
        assert len(inputs) == 3
        return [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]

    async def fake_native_rerank(question, chunks, config):
        del question, config
        return [0.2, 0.8][: len(chunks)]

    monkeypatch.setattr(service, "_embeddings", fake_embeddings)
    monkeypatch.setattr(service, "_native_rerank", fake_native_rerank)
    chunks = [{"text": "相关", "page": 1}, {"text": "无关", "page": 2}]

    scores, protocol = asyncio.run(service._retrieve_scores("查询", chunks, _config("embedding")))
    rerank_scores, rerank_protocol = asyncio.run(service._rerank_scores("查询", chunks, _config("reranker")))

    assert scores[0] > scores[1]
    assert protocol == "openai_embeddings"
    assert rerank_scores == [0.2, 0.8]
    assert rerank_protocol == "compatible_rerank"


def test_reranker_failure_preserves_retrieval_scores_without_chat_fallback(monkeypatch) -> None:
    service = RagRetrievalService(_IndexStub())

    async def reject_native(question, chunks, config):
        del question, chunks, config
        raise AppError("MODEL_REQUEST_REJECTED", "unsupported", 502)

    async def chat_must_not_run(*args, **kwargs):
        del args, kwargs
        raise AssertionError("reranker degradation must preserve retrieval order without another model call")

    monkeypatch.setattr(service, "_native_rerank", reject_native)
    monkeypatch.setattr(service, "_chat_scores", chat_must_not_run)

    scores, protocol = asyncio.run(
        service._rerank_scores(
            "查询",
            [{"text": "候选一", "retrieval_score": 0.35}, {"text": "候选二", "retrieval_score": 0.85}],
            _config("reranker"),
        )
    )

    assert scores == [0.35, 0.85]
    assert protocol == "retrieval_score_fallback"


def test_embedding_rejection_falls_back_to_local_keyword_scores(monkeypatch) -> None:
    service = RagRetrievalService(_IndexStub())

    async def reject_embeddings(inputs, config):
        del inputs, config
        raise AppError("MODEL_REQUEST_REJECTED", "unsupported", 502)

    async def chat_must_not_run(*args, **kwargs):
        del args, kwargs
        raise AssertionError("embedding degradation must be local keyword-only")

    monkeypatch.setattr(service, "_embeddings", reject_embeddings)
    monkeypatch.setattr(service, "_chat_scores", chat_must_not_run)
    scores, protocol = asyncio.run(
        service._retrieve_scores(
            "偏心锁紧楔块",
            [{"id": "exact", "text": "偏心锁紧楔块用于压紧工件"}, {"id": "noise", "text": "普通背景说明"}],
            _config("chat"),
        )
    )

    assert scores[0] > scores[1]
    assert protocol == "keyword_only_degraded"


def test_retrieval_connection_prefers_embeddings_and_reports_protocol(monkeypatch) -> None:
    service = RagRetrievalService(_IndexStub())

    async def fake_embeddings(inputs, config):
        del config
        assert inputs == ["工业文档检索连通测试"]
        return [[0.1, 0.2, 0.3]]

    monkeypatch.setattr(service, "_embeddings", fake_embeddings)
    result = asyncio.run(service.test_connection("retrieval", _config("embedding")))

    assert result["status"] == "connected"
    assert result["protocol"] == "openai_embeddings"
    assert result["dimensions"] == 3


def test_ocr_corpus_cache_is_file_backed_and_never_contains_api_keys(tmp_path) -> None:
    document = tmp_path / "sample.pdf"
    document.write_bytes(b"%PDF-cache-fingerprint")
    index = DocumentChunkIndex(tmp_path / "rag-cache")

    index.save_ocr_cache(document, "ocr-model", {2: "识别后的部件说明"})
    loaded = DocumentChunkIndex(tmp_path / "rag-cache").load_ocr_cache(document, "ocr-model")

    assert loaded == {2: "识别后的部件说明"}
    cache_text = "\n".join(path.read_text(encoding="utf-8") for path in (tmp_path / "rag-cache").glob("*.json"))
    assert "识别后的部件说明" in cache_text
    assert "api_key" not in cache_text


def test_specialized_endpoint_url_accepts_root_v1_and_full_chat_urls() -> None:
    assert _endpoint_url("https://models.example", "embeddings") == "https://models.example/v1/embeddings"
    assert _endpoint_url("https://models.example/v1", "embeddings") == "https://models.example/v1/embeddings"
    assert (
        _endpoint_url("https://models.example/v1/chat/completions", "embeddings")
        == "https://models.example/v1/embeddings"
    )
    assert LangChainFaissIndex._openai_root("https://models.example/v1/embeddings") == "https://models.example/v1"


def test_langchain_faiss_snapshot_is_safe_reusable_and_split_ids_are_unique(tmp_path, monkeypatch) -> None:
    import faiss
    import numpy as np
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    class FakeEmbeddings:
        document_calls = 0

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        @staticmethod
        def _vector(text):
            return [float(len(text)), float(text.count("定位") + 1), float(text.count("挡板") + 1)]

        def embed_documents(self, texts):
            type(self).document_calls += 1
            return [self._vector(text) for text in texts]

        def embed_query(self, text):
            return self._vector(text)

    monkeypatch.setattr(
        LangChainFaissIndex,
        "_components",
        staticmethod(lambda: (Document, RecursiveCharacterTextSplitter, FakeEmbeddings, faiss, np)),
    )
    document = tmp_path / "sample.pdf"
    document.write_bytes(b"%PDF-enterprise-index")
    manager = LangChainFaissIndex(tmp_path / "faiss")
    chunks = DocumentChunkIndex(chunk_size=80, overlap=10).chunk_pages(
        [{"page": 1, "source": "ocr", "text": ("定位挡板用于固定工件。驱动件推动挡板完成定位。" * 20)}]
    )

    first, first_status = manager._search_sync(
        "如何定位工件", chunks, document, _config("embedding-model"), "ocr-model", 80, 10, 5
    )
    second, second_status = manager._search_sync(
        "如何定位工件", chunks, document, _config("embedding-model"), "ocr-model", 80, 10, 5
    )

    assert first and second
    assert first_status["vector_index_cache"] == "built"
    assert second_status["vector_index_cache"] == "hit"
    assert FakeEmbeddings.document_calls == 1
    snapshot = tmp_path / "faiss" / first_status["vector_index_key"]
    assert {path.name for path in snapshot.iterdir()} == {"vectors.npy", "documents.json", "manifest.json"}
    persisted_documents = __import__("json").loads((snapshot / "documents.json").read_text(encoding="utf-8"))
    persisted_ids = [item["id"] for item in persisted_documents]
    assert len(persisted_ids) > 1
    assert len(persisted_ids) == len(set(persisted_ids))
    matrix = np.load(snapshot / "vectors.npy", allow_pickle=False)
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    assert matrix.dtype == np.float32
    assert matrix.shape == (manifest["vector_count"], manifest["vector_dimension"])
    assert not (snapshot / "index.pkl").exists()
    assert not (snapshot / "index.faiss").exists()
    assert not list(snapshot.glob("*.pkl"))


def test_faiss_manifest_key_invalidates_when_pdf_fingerprint_changes(tmp_path) -> None:
    document = tmp_path / "sample.pdf"
    document.write_bytes(b"first")
    manager = LangChainFaissIndex(tmp_path / "faiss")
    chunks = [{"id": "p1-c1", "page": 1, "source": "pdf_text", "text": "定位说明"}]

    first_key, _ = manager._manifest(
        chunks, document, _config("embedding"), ocr_model="ocr", chunk_size=900, overlap=120
    )
    document.write_bytes(b"second-version")
    second_key, _ = manager._manifest(
        chunks, document, _config("embedding"), ocr_model="ocr", chunk_size=900, overlap=120
    )

    assert first_key != second_key


def test_missing_vector_dependencies_are_reported_as_explicit_degradation(monkeypatch) -> None:
    manager = LangChainFaissIndex()

    def missing_components():
        raise AppError("RAG_VECTOR_DEPENDENCY_MISSING", "missing", 503, False)

    monkeypatch.setattr(manager, "_components", missing_components)
    try:
        manager._search_sync(
            "query",
            [{"id": "1", "page": 1, "source": "ocr", "text": "content"}],
            Path(__file__),
            _config("embedding"),
            "ocr",
            900,
            120,
            1,
        )
    except AppError as exc:
        assert exc.code == "RAG_VECTOR_DEPENDENCY_MISSING"
    else:  # pragma: no cover
        raise AssertionError("missing dependency must not be presented as FAISS success")


def _patent_sections() -> list[str]:
    return [
        "技术领域\n本发明涉及工业工件定位装置。",
        "背景技术\n现有装置在高速输送时容易发生横向偏移。",
        "发明内容\n通过限位挡板与导向槽配合保持工件稳定。",
        "具体实施方式\n驱动件推动偏心锁紧楔块压紧工件。",
        "权利要求1\n一种包括限位挡板和导向槽的定位装置。",
    ]


def test_patent_chunk_ids_are_deterministic_unique_and_content_bound() -> None:
    index = DocumentChunkIndex(chunk_size=72, overlap=8)
    sections = _patent_sections()
    base = index.chunk_pages([{"page": 7, "source": "pdf_text", "text": "\n\n".join(sections)}])
    repeated = index.chunk_pages([{"page": 7, "source": "pdf_text", "text": "\n\n".join(sections)}])

    assert base == repeated
    assert len({item["id"] for item in base}) == len(base)
    for item in base:
        digest = __import__("hashlib").sha256(item["text"].encode("utf-8")).hexdigest()[:8]
        assert item["id"].endswith(digest)


def test_patent_structure_chunks_keep_boundaries_and_ids_stable_after_prefix_insertion() -> None:
    index = DocumentChunkIndex(chunk_size=72, overlap=8)
    sections = _patent_sections()
    base = index.chunk_pages([{"page": 7, "source": "pdf_text", "text": "\n\n".join(sections)}])
    prefixed = index.chunk_pages(
        [{"page": 7, "source": "pdf_text", "text": "摘要\n用于工业定位。\n\n" + "\n\n".join(sections)}]
    )

    assert all(any(item["text"].startswith(heading) for item in base) for heading in ("技术领域", "背景技术", "发明内容", "具体实施方式", "权利要求1"))
    base_by_text = {item["text"]: item["id"] for item in base}
    prefixed_by_text = {item["text"]: item["id"] for item in prefixed}
    unchanged = set(base_by_text) & set(prefixed_by_text)
    assert unchanged
    assert all(base_by_text[text] == prefixed_by_text[text] for text in unchanged)


class _HybridSearchIndex:
    def __init__(self, chunks, cache_dir: Path) -> None:
        self._chunks = chunks
        self.cache_dir = cache_dir
        self.chunk_size = 720
        self.overlap = 100

    def pages(self, path):
        del path
        return [
            {"page": item["page"], "text": item["text"], "source": item.get("source", "pdf_text")}
            for item in self._chunks
        ]

    def load_ocr_cache(self, path, model):
        del path, model
        return {}

    def save_ocr_cache(self, path, model, pages):
        del path, model, pages

    def chunk_pages(self, pages):
        return [
            {
                "id": self._chunks[index]["id"],
                "page": page["page"],
                "text": page["text"],
                "source": page.get("source", "pdf_text"),
            }
            for index, page in enumerate(pages)
        ]


class _VectorSearchStub:
    def __init__(self, items=None, error: AppError | None = None) -> None:
        self.items = items or []
        self.error = error

    async def search(self, *args, **kwargs):
        del args, kwargs
        if self.error:
            raise self.error
        return [dict(item) for item in self.items], {
            "vector_index_status": "ready",
            "vector_index_cache": "hit",
            "vector_index_key": "test-vector",
            "vector_persistence": "vectors.npy",
            "degradation_reason": None,
        }


def _hybrid_chunks():
    return [
        {"id": "exact", "page": 1, "source": "pdf_text", "text": "偏心锁紧楔块用于压紧工件并限制横向移动。"},
        {"id": "semantic", "page": 2, "source": "pdf_text", "text": "该机构能够对输送中的零件进行稳定定位。"},
        {"id": "noise", "page": 3, "source": "pdf_text", "text": "背景技术介绍了常见驱动电机。"},
    ]


def test_keyword_and_vector_candidates_are_fused_before_reranking(tmp_path, monkeypatch) -> None:
    chunks = _hybrid_chunks()
    vector = _VectorSearchStub(
        [
            {**chunks[1], "vector_score": 0.98},
            {**chunks[2], "vector_score": 0.70},
        ]
    )
    service = RagRetrievalService(_HybridSearchIndex(chunks, tmp_path / "rag"), vector)

    async def preserve_fusion(question, items, config):
        del question, config
        return [float(item["retrieval_score"]) for item in items]

    monkeypatch.setattr(service, "_native_rerank", preserve_fusion)
    result = asyncio.run(
        service.search(
            "偏心锁紧楔块如何压紧工件？",
            tmp_path / "sample.pdf",
            _config("embedding"),
            _config("reranker"),
            _config("ocr"),
            target_page=1,
            top_k=3,
        )
    )

    assert result["items"][0]["id"] == "exact"
    assert {item["id"] for item in result["items"]} >= {"exact", "semantic"}
    assert all("retrieval_score" in item for item in result["items"])
    by_id = {item["id"]: item for item in result["items"]}
    assert by_id["exact"]["keyword_score"] > 0
    assert by_id["semantic"]["vector_score"] == 0.98
    assert result["retrieval_protocol"] == "hybrid_keyword_vector_npy"


def test_vector_failure_degrades_to_keyword_only_without_external_retrieval_call(tmp_path, monkeypatch) -> None:
    chunks = _hybrid_chunks()
    vector = _VectorSearchStub(error=AppError("RAG_VECTOR_INDEX_UNAVAILABLE", "embedding down", 502, True))
    service = RagRetrievalService(_HybridSearchIndex(chunks, tmp_path / "rag"), vector)

    async def external_retrieval_must_not_run(*args, **kwargs):
        del args, kwargs
        raise AssertionError("keyword-only degradation must not call Retrieval Chat Completions")

    async def preserve_order(question, items, config):
        del question, config
        return [float(item["retrieval_score"]) for item in items]

    monkeypatch.setattr(service, "_chat_scores", external_retrieval_must_not_run)
    monkeypatch.setattr(service, "_native_rerank", preserve_order)
    result = asyncio.run(
        service.search(
            "偏心锁紧楔块",
            tmp_path / "sample.pdf",
            _config("embedding"),
            _config("reranker"),
            _config("ocr"),
            target_page=1,
        )
    )

    assert result["items"][0]["id"] == "exact"
    assert result["vector_index_status"] == "degraded"
    assert "keyword" in result["retrieval_protocol"]
    assert result["degradation_reason"]["code"] == "RAG_VECTOR_INDEX_UNAVAILABLE"


def test_reranker_failure_returns_fused_order_instead_of_raising(tmp_path, monkeypatch) -> None:
    chunks = _hybrid_chunks()
    vector = _VectorSearchStub([{**chunks[1], "vector_score": 0.98}, {**chunks[0], "vector_score": 0.70}])
    service = RagRetrievalService(_HybridSearchIndex(chunks, tmp_path / "rag"), vector)

    async def fail_reranker(question, items, config):
        del question, items, config
        raise AppError("MODEL_TIMEOUT", "reranker timeout", 504, True)

    monkeypatch.setattr(service, "_native_rerank", fail_reranker)
    result = asyncio.run(
        service.search(
            "偏心锁紧楔块如何压紧工件？",
            tmp_path / "sample.pdf",
            _config("embedding"),
            _config("reranker"),
            _config("ocr"),
            target_page=1,
            top_k=3,
        )
    )

    assert result["items"]
    assert [item["retrieval_score"] for item in result["items"]] == sorted(
        [item["retrieval_score"] for item in result["items"]], reverse=True
    )
    assert result["rerank_protocol"] == "retrieval_score_fallback"
    assert result["rerank_degradation_reason"]["code"] == "MODEL_TIMEOUT"


def test_offline_paginated_ocr_txt_is_used_before_online_ocr(tmp_path, monkeypatch) -> None:
    document = tmp_path / "semi_final_dataset" / "train" / "documents" / "CN-DEMO.pdf"
    document.parent.mkdir(parents=True)
    document.write_bytes(b"%PDF-offline-corpus")
    corpus_dir = tmp_path / "runtime" / "ocr-corpus" / "train" / "CN-DEMO"
    corpus_dir.mkdir(parents=True)
    (corpus_dir / "page-0001.txt").write_text("离线 OCR：偏心锁紧楔块用于压紧工件。", encoding="utf-8")
    stat = document.stat()
    fingerprint = hashlib.sha256(
        f"{document.name}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")
    ).hexdigest()
    (corpus_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "document": "CN-DEMO.pdf",
                "source": {
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "fingerprint": fingerprint,
                },
                "ocr": {"model": "ocr", "endpoint_hash": "test-endpoint"},
                "page_count": 1,
                "completed_pages": 1,
                "failed_pages": 0,
                "pages": {"1": {"status": "completed", "attempts": 1, "file": "page-0001.txt", "error": None}},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    class OfflineIndex(DocumentChunkIndex):
        def pages(self, path):
            del path
            return [{"page": 1, "text": ""}]

    monkeypatch.setattr(retrieval_module, "ROOT_DIR", tmp_path)
    index = OfflineIndex(tmp_path / "runtime" / "rag-index")
    service = RagRetrievalService(index, _VectorSearchStub([]))

    async def online_ocr_must_not_run(*args, **kwargs):
        del args, kwargs
        raise AssertionError("offline OCR page must take precedence over online OCR")

    monkeypatch.setattr(retrieval_module.OpenAICompatibleClient, "chat", online_ocr_must_not_run)
    chunks, metadata = asyncio.run(
        service._hybrid_corpus(
            document,
            _config("ocr"),
            target_page=1,
            target_ocr="",
            render_page=lambda page: f"page-{page}".encode(),
        )
    )

    assert chunks[0]["text"] == "离线 OCR：偏心锁紧楔块用于压紧工件。"
    assert chunks[0]["source"] == "ocr"
    assert metadata["ocr_pages"] == [1]
    assert metadata["precomputed_ocr_pages"] == [1]


def test_precompute_ocr_script_declares_paginated_atomic_corpus_contract() -> None:
    script = Path(__file__).resolve().parent.parent / "scripts" / "precompute_semifinal_ocr.py"
    if not script.is_file():
        pytest.xfail("scripts/precompute_semifinal_ocr.py 尚未落地")
    source = script.read_text(encoding="utf-8")

    for function_name in ("load_ocr_config", "discover_documents", "build_plan", "run"):
        assert f"def {function_name}" in source or f"async def {function_name}" in source
    for contract in ("runtime", "ocr-corpus", "page-", "manifest.json", "completed_pages", "failed_pages"):
        assert contract in source
    assert "api_key" not in source[source.find("manifest"): source.find("manifest") + 1000].lower()


def _write_corpus(tmp_path, document, model="ocr-model", fingerprint=None, with_manifest=True):
    corpus_dir = tmp_path / "runtime" / "ocr-corpus" / "train" / document.stem
    corpus_dir.mkdir(parents=True, exist_ok=True)
    (corpus_dir / "page-0001.txt").write_text("离线 OCR 文本。", encoding="utf-8")
    if not with_manifest:
        return corpus_dir
    stat = document.stat()
    digest = fingerprint or hashlib.sha256(
        f"{document.name}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")
    ).hexdigest()
    (corpus_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "document": document.name,
                "source": {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "fingerprint": digest},
                "ocr": {"model": model, "endpoint_hash": "endpoint"},
                "page_count": 1,
                "completed_pages": 1,
                "failed_pages": 0,
                "pages": {"1": {"status": "completed", "file": "page-0001.txt"}},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return corpus_dir


def test_txt_ocr_corpus_is_accepted_when_model_and_fingerprint_match(tmp_path, monkeypatch) -> None:
    document = tmp_path / "semi_final_dataset" / "train" / "documents" / "CN-A.pdf"
    document.parent.mkdir(parents=True)
    document.write_bytes(b"%PDF-a")
    _write_corpus(tmp_path, document)
    monkeypatch.setattr(retrieval_module, "ROOT_DIR", tmp_path)

    corpus = retrieval_module.DocumentChunkIndex.load_txt_ocr_corpus(document, "ocr-model")

    assert corpus["pages"] == {1: "离线 OCR 文本。"}
    assert corpus["meta"]["stale"] is False
    assert corpus["meta"]["model"] == "ocr-model"
    assert corpus["meta"]["model_match"] is True
    assert corpus["meta"]["fingerprint_match"] is True


def test_txt_ocr_corpus_is_rejected_when_ocr_model_changed(tmp_path, monkeypatch) -> None:
    document = tmp_path / "semi_final_dataset" / "train" / "documents" / "CN-B.pdf"
    document.parent.mkdir(parents=True)
    document.write_bytes(b"%PDF-b")
    _write_corpus(tmp_path, document, model="old-ocr-model")
    monkeypatch.setattr(retrieval_module, "ROOT_DIR", tmp_path)

    corpus = retrieval_module.DocumentChunkIndex.load_txt_ocr_corpus(document, "new-ocr-model")

    assert corpus["pages"] == {}
    assert corpus["meta"]["stale"] is True
    assert corpus["meta"]["model_match"] is False
    assert "OCR 模型已从" in corpus["meta"]["stale_reason"]


def test_txt_ocr_corpus_is_rejected_when_pdf_fingerprint_changed(tmp_path, monkeypatch) -> None:
    document = tmp_path / "semi_final_dataset" / "train" / "documents" / "CN-C.pdf"
    document.parent.mkdir(parents=True)
    document.write_bytes(b"%PDF-c")
    _write_corpus(tmp_path, document, fingerprint="0" * 64)
    monkeypatch.setattr(retrieval_module, "ROOT_DIR", tmp_path)

    corpus = retrieval_module.DocumentChunkIndex.load_txt_ocr_corpus(document, "ocr-model")

    assert corpus["pages"] == {}
    assert corpus["meta"]["stale"] is True
    assert "指纹" in corpus["meta"]["stale_reason"]


def test_txt_ocr_corpus_without_manifest_is_not_silently_trusted(tmp_path, monkeypatch) -> None:
    document = tmp_path / "semi_final_dataset" / "train" / "documents" / "CN-D.pdf"
    document.parent.mkdir(parents=True)
    document.write_bytes(b"%PDF-d")
    _write_corpus(tmp_path, document, with_manifest=False)
    monkeypatch.setattr(retrieval_module, "ROOT_DIR", tmp_path)

    corpus = retrieval_module.DocumentChunkIndex.load_txt_ocr_corpus(document, "ocr-model")

    assert corpus["pages"] == {}
    assert corpus["meta"]["stale"] is True
    assert "manifest" in corpus["meta"]["stale_reason"]


def test_txt_ocr_corpus_audit_view_can_expose_stale_pages_with_labels(tmp_path, monkeypatch) -> None:
    document = tmp_path / "semi_final_dataset" / "train" / "documents" / "CN-E.pdf"
    document.parent.mkdir(parents=True)
    document.write_bytes(b"%PDF-e")
    _write_corpus(tmp_path, document, model="old-ocr-model")
    monkeypatch.setattr(retrieval_module, "ROOT_DIR", tmp_path)

    corpus = retrieval_module.DocumentChunkIndex.load_txt_ocr_corpus(
        document, "new-ocr-model", allow_model_mismatch=True
    )

    assert corpus["pages"] == {1: "离线 OCR 文本。"}
    assert corpus["meta"]["stale"] is True
    assert corpus["meta"]["model"] == "old-ocr-model"
    assert corpus["meta"]["expected_model"] == "new-ocr-model"
