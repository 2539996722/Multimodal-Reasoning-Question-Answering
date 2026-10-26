from __future__ import annotations

import asyncio
import importlib.util
import json
import os
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Query, Request
from fastapi.responses import FileResponse, Response, StreamingResponse

from app.config import ROOT_DIR, settings_store
from app.errors import AppError
from app.models.schemas import AnalyzeRequest, JobCreateRequest, ModelSettingsUpdate, ModelTestRequest, Split, Stage
from app.services.analysis import AnalysisPipeline, RunStore
from app.services.dataset import DatasetRepository
from app.services.documents import DocumentService
from app.services.export import ExportService
from app.services.jobs import JobManager
from app.services.openai_client import OpenAICompatibleClient
from app.services.retrieval import RagRetrievalService


router = APIRouter(prefix="/api")
repository = DatasetRepository(ROOT_DIR)
document_service = DocumentService()
run_store = RunStore(max_items=100)
analysis_pipeline = AnalysisPipeline(repository, document_service, settings_store, run_store)
job_manager = JobManager(repository, analysis_pipeline, settings_store.settings)
export_service = ExportService()
rag_service = RagRetrievalService()


def _rag_runtime() -> dict:
    runtime_dir = ROOT_DIR / "runtime"
    return {
        **rag_service.capabilities(),
        "cache_writable": runtime_dir.is_dir() and os.access(runtime_dir, os.W_OK),
    }


@router.get("/health")
async def health() -> dict:
    datasets = repository.datasets()
    total_questions = sum(item["question_count"] for item in datasets["items"])
    return {
        "status": "ok" if total_questions else "degraded",
        "service": "InduDoc AI",
        "version": settings_store.settings.version,
        "dataset_available": total_questions > 0,
        "dataset_question_count": total_questions,
        "models": settings_store.public(),
        "rag": _rag_runtime(),
    }


@router.get("/health/live")
async def liveness() -> dict:
    """Cheap process liveness probe. It never calls an external model."""
    return {"status": "ok", "service": "InduDoc AI", "version": settings_store.settings.version}


@router.get("/health/ready")
async def readiness() -> dict:
    """Deployment readiness based on local assets and RAG runtime dependencies."""
    datasets = repository.datasets()
    total_questions = sum(item["question_count"] for item in datasets["items"])
    rag = _rag_runtime()
    checks = {
        "dataset": total_questions > 0,
        "langchain": rag["langchain_available"],
        "faiss": rag["faiss_available"],
        "runtime_cache": rag["cache_writable"],
    }
    ready = all(checks.values())
    return {
        "status": "ready" if ready else "degraded",
        "ready": ready,
        "checks": checks,
        "models_configured": {
            kind: settings_store.endpoint(kind).configured
            for kind in ("ocr", "vlm", "retrieval", "reranker")
        },
    }


@router.get("/datasets")
async def datasets() -> dict:
    return repository.datasets()


@router.get("/questions")
async def questions(
    stage: Stage,
    split: Split,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 30,
    q: str | None = None,
    keyword: str | None = None,
    document: str | None = None,
    id: str | None = None,
    question_id: str | None = None,
) -> dict:
    return repository.query_questions(
        stage,
        split,
        page,
        page_size,
        keyword=q or keyword,
        document=document,
        question_id=id or question_id,
    )


@router.get("/questions/{stage}/{split}/{question_id}")
async def question_detail(stage: Stage, split: Split, question_id: str) -> dict:
    return repository.get_question(stage, split, question_id).model_dump()


@router.get("/documents/{stage}/{split}/{filename}")
async def document_file(stage: Stage, split: Split, filename: str, download: bool = False) -> FileResponse:
    path = repository.resolve_document(stage, split, filename)
    disposition = "attachment" if download else "inline"
    return FileResponse(path, media_type="application/pdf", filename=path.name, content_disposition_type=disposition)


@router.get("/documents/{stage}/{split}/{filename}/meta")
async def document_meta(stage: Stage, split: Split, filename: str) -> dict:
    path = repository.resolve_document(stage, split, filename)
    return {"stage": stage, "split": split, **document_service.metadata(path)}


@router.get("/documents/{stage}/{split}/{filename}/ocr")
async def document_ocr(stage: Stage, split: Split, filename: str) -> dict:
    path = repository.resolve_document(stage, split, filename)
    corpus = rag_service.index.load_txt_ocr_corpus(
        path,
        settings_store.endpoint("ocr").model,
        allow_model_mismatch=True,
    )
    pages = corpus.get("pages") or {}
    corpus_meta = corpus.get("meta") or {}
    total_pages = int(document_service.metadata(path).get("page_count") or len(pages))
    page_items = [
        {
            "page": page,
            "cached": page in pages,
            "characters": len(pages.get(page, "")),
            "text": pages.get(page, ""),
        }
        for page in range(1, total_pages + 1)
    ]
    text = "\n\n".join(f"===== PDF P.{page} =====\n{pages[page].strip()}" for page in sorted(pages))
    return {
        "stage": stage,
        "split": split,
        "document": path.name,
        "cache_hit": bool(pages),
        "pages": sorted(pages),
        "page_count": len(pages),
        "total_pages": total_pages,
        "page_items": page_items,
        "characters": len(text),
        "text": text,
        "corpus": corpus_meta,
    }


@router.get("/documents/{stage}/{split}/{filename}/pages/{page}")
async def document_page(
    stage: Stage,
    split: Split,
    filename: str,
    page: int,
    scale: Annotated[float, Query(ge=0.75, le=3.0)] = 1.6,
) -> Response:
    path = repository.resolve_document(stage, split, filename)
    image = document_service.render_page(path, page, scale)
    return Response(image, media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})


@router.post("/analyze")
async def analyze(payload: AnalyzeRequest) -> dict:
    return await analysis_pipeline.analyze(payload)


@router.post("/analyze/stream")
async def analyze_stream(payload: AnalyzeRequest, request: Request) -> StreamingResponse:
    """Stream actual pipeline stage transitions, then emit the final analysis response."""

    async def event_source():
        queue: asyncio.Queue[dict] = asyncio.Queue()

        def stage_callback(stage: dict) -> None:
            queue.put_nowait({"event": "stage", "data": stage})

        task = asyncio.create_task(analysis_pipeline.analyze(payload, event_callback=stage_callback))
        try:
            while not task.done() or not queue.empty():
                if await request.is_disconnected():
                    task.cancel()
                    return
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=0.5)
                except TimeoutError:
                    continue
                yield f"event: {item['event']}\ndata: {json.dumps(item['data'], ensure_ascii=False)}\n\n"
            try:
                result = task.result()
            except AppError as exc:
                error = {
                    "code": exc.code,
                    "message": exc.message,
                    "request_id": request.state.request_id,
                    "retryable": exc.retryable,
                }
                yield f"event: error\ndata: {json.dumps(error, ensure_ascii=False)}\n\n"
            except Exception:
                error = {
                    "code": "INTERNAL_ERROR",
                    "message": "服务处理分析任务时发生内部错误",
                    "request_id": request.state.request_id,
                    "retryable": True,
                }
                yield f"event: error\ndata: {json.dumps(error, ensure_ascii=False)}\n\n"
            else:
                yield f"event: result\ndata: {json.dumps(result, ensure_ascii=False)}\n\n"
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/jobs", status_code=202)
async def create_job(payload: JobCreateRequest) -> dict:
    return await job_manager.create(payload)


@router.get("/jobs")
async def jobs(limit: Annotated[int, Query(ge=1, le=100)] = 50) -> dict:
    items = job_manager.list(limit)
    return {"items": items, "total": len(items)}


@router.get("/jobs/{job_id}")
async def job_detail(job_id: str) -> dict:
    return job_manager.get(job_id)


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str) -> dict:
    return await job_manager.cancel(job_id)


@router.post("/jobs/{job_id}/retry", status_code=202)
async def retry_failed_job_items(job_id: str) -> dict:
    return await job_manager.retry_failed(job_id)


@router.get("/jobs/{job_id}/export")
async def export_job(job_id: str) -> FileResponse:
    job = job_manager.get(job_id)
    results = job_manager.results(job_id)
    if not results:
        raise AppError("JOB_RESULTS_EMPTY", "任务尚无可导出的结果", 409)
    path = export_service.write_jsonl(job["stage"], results)
    return FileResponse(path, media_type="application/x-ndjson", filename=path.name)


@router.get("/jobs/{job_id}/summary")
async def job_summary(job_id: str) -> dict:
    job = job_manager.get(job_id)
    return {key: value for key, value in job.items() if key != "results"}


@router.get("/runs")
async def runs(limit: Annotated[int, Query(ge=1, le=100)] = 100) -> dict:
    items = run_store.list(limit)
    return {"items": items, "total": len(items)}


@router.get("/settings/models")
async def model_settings() -> dict:
    return settings_store.public()


@router.put("/settings/models")
async def update_model_settings(payload: ModelSettingsUpdate) -> dict:
    return settings_store.update(payload.model_dump(exclude_none=True))


@router.post("/settings/models/test")
async def test_model_settings(payload: ModelTestRequest) -> dict:
    selected = payload.selected_service
    kinds = (
        ("ocr", "vlm", "retrieval", "reranker")
        if selected == "all"
        else ("ocr", "vlm")
        if selected == "both"
        else (selected,)
    )
    results = []
    for kind in kinds:
        if kind in {"retrieval", "reranker"}:
            result = await rag_service.test_connection(kind, settings_store.endpoint(kind))
        else:
            result = await OpenAICompatibleClient(settings_store.endpoint(kind), kind).test_connection()
        results.append(result)
    return {"status": "connected", "items": results}
