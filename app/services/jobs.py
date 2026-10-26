from __future__ import annotations

import asyncio
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from app.config import AppSettings
from app.errors import AppError
from app.models.schemas import AnalyzeRequest, JobCreateRequest
from app.services.analysis import AnalysisPipeline
from app.services.dataset import DatasetRepository


STAGE_PLANS: dict[str, tuple[tuple[str, float], ...]] = {
    "preliminary": (
        ("locating", 0.12),
        ("ocr", 0.35),
        ("reasoning", 0.43),
        ("validation", 0.10),
    ),
    "semi_final": (
        ("locating", 0.08),
        ("ocr", 0.14),
        ("indexing", 0.22),
        ("retrieval", 0.12),
        ("reranking", 0.10),
        ("reasoning", 0.28),
        ("validation", 0.06),
    ),
}

STAGE_DEFAULTS: dict[str, tuple[str, str]] = {
    "queued": ("等待执行槽位", "AsyncIO bounded semaphore"),
    "locating": ("文档定位与页面渲染", "DatasetRepository + PyMuPDF"),
    "ocr": ("OCR 文档识别", "OpenAI-compatible OCR"),
    "indexing": ("RAG 混合语料索引", "OCR-aware document index"),
    "retrieval": ("向量召回", "OpenAI-compatible Embeddings + FAISS"),
    "reranking": ("候选精排", "OpenAI-compatible Rerank API"),
    "reasoning": ("VLM 图文联合推理", "OpenAI-compatible multimodal model"),
    "validation": ("答案规范化与评测", "Answer Normalizer + Metrics"),
}

PUBLIC_STAGE_KEYS = {
    "name",
    "status",
    "title",
    "detail",
    "technology",
    "duration_ms",
    "model",
    "output_summary",
}
SECRET_KEYS = {"api_key", "authorization", "access_token", "refresh_token", "secret"}


class JobManager:
    def __init__(
        self,
        repository: DatasetRepository,
        pipeline: AnalysisPipeline,
        app_settings: AppSettings,
    ) -> None:
        self.repository = repository
        self.pipeline = pipeline
        self.app_settings = app_settings
        self._jobs: dict[str, dict[str, Any]] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._lock = asyncio.Lock()

    async def create(self, request: JobCreateRequest) -> dict[str, Any]:
        if request.limit > self.app_settings.max_batch_size:
            raise AppError("BATCH_LIMIT_EXCEEDED", f"单次批量最多 {self.app_settings.max_batch_size} 题", 400)
        if request.concurrency > self.app_settings.max_concurrency:
            raise AppError("CONCURRENCY_LIMIT_EXCEEDED", f"并发数最多 {self.app_settings.max_concurrency}", 400)
        partition = self.repository.partition(request.stage, request.split)
        if request.question_ids:
            selected = [self.repository.get_question(request.stage, request.split, item) for item in request.question_ids]
            selected = selected[: request.limit]
        else:
            selected = partition.records[request.offset : request.offset + request.limit]
        if not selected:
            raise AppError("EMPTY_JOB", "所选范围没有题目", 400)
        job_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        # Batch evaluation is a real-model operation by default.  Global demo
        # mode is intentionally ignored here; only an explicit true opt-in may
        # replay references.
        force_demo = request.force_demo is True
        normalized_request = request.model_dump()
        normalized_request["force_demo"] = force_demo
        items = [self._new_item(record, position) for position, record in enumerate(selected, start=1)]
        job: dict[str, Any] = {
            "job_id": job_id,
            "status": "queued",
            "stage": request.stage,
            "split": request.split,
            "mode": "demo" if force_demo else "real",
            "use_ocr": request.use_ocr if request.stage == "preliminary" else True,
            "use_rag": request.use_rag if request.stage == "semi_final" else False,
            "concurrency": request.concurrency,
            "total": len(selected),
            "completed": 0,
            "success": 0,
            "failed": 0,
            "progress": 0.0,
            "success_rate": 0.0,
            "error_code_counts": {},
            "current_question_ids": [],
            "current_stage": "queued",
            "stage_summary": {},
            "items": items,
            "average_duration_ms": None,
            "metrics": {
                "applicable": False,
                "accuracy": None,
                "exact_match": None,
                "rouge1_f1": None,
                "semantic_f1": None,
                "bert_f1": None,
                "final_score": None,
            },
            "results": [],
            "created_at": now,
            "started_at": None,
            "finished_at": None,
            "cancel_requested": False,
            "request": normalized_request,
        }
        self._refresh_job_state(job)
        async with self._lock:
            self._jobs[job_id] = job
            self._tasks[job_id] = asyncio.create_task(self._run(job, selected, request, force_demo))
        return self._public(job, include_results=False)

    async def _run(
        self,
        job: dict[str, Any],
        selected: list[Any],
        request: JobCreateRequest,
        force_demo: bool,
    ) -> None:
        job["status"] = "running"
        job["started_at"] = datetime.now(timezone.utc).isoformat()
        semaphore = asyncio.Semaphore(request.concurrency)

        async def process(record: Any, item: dict[str, Any]) -> None:
            if job["cancel_requested"]:
                return
            async with semaphore:
                if job["cancel_requested"]:
                    return
                self._set_item_running(item)
                job["current_question_ids"].append(record.id)
                self._refresh_job_state(job)
                started = time.perf_counter()
                terminal = False

                def stage_callback(stage: dict[str, Any]) -> None:
                    self._update_item_stage(item, stage)
                    self._refresh_job_state(job)

                try:
                    response = await self.pipeline.analyze(
                        AnalyzeRequest(
                            stage=request.stage,
                            split=request.split,
                            question_id=record.id,
                            force_demo=force_demo,
                            use_ocr=request.use_ocr,
                            use_rag=request.use_rag,
                            evaluate_semantic=request.evaluate_semantic,
                        ),
                        source="batch_item",
                        event_callback=stage_callback,
                    )
                    result = response["result"]
                    job["results"].append(
                        {
                            "id": record.id,
                            "status": "completed",
                            "answer": result["answer"],
                            "result": result,
                            "run_id": response["run_id"],
                        }
                    )
                    item["status"] = "completed"
                    item["stage_status"] = "completed"
                    item["answer"] = result["answer"]
                    item["run_id"] = response["run_id"]
                    # Keep the batch console auditable without exposing hidden
                    # chain-of-thought or provider internals.
                    for key in ("answer_explanation", "reasoning_summary", "evidence", "option_analysis", "ocr_pages"):
                        if result.get(key) is not None:
                            item[key] = result[key]
                    self._complete_unvisited_stages(item, job["stage"])
                    job["success"] += 1
                    terminal = True
                except AppError as exc:
                    self._record_item_failure(job, item, record.id, exc.code, exc.message, exc.retryable)
                    terminal = True
                except asyncio.CancelledError:
                    self._record_item_failure(
                        job, item, record.id, "JOB_CANCELLED", "任务已取消", True
                    )
                    terminal = True
                    raise
                except Exception:
                    # Never echo a provider exception into the public job
                    # payload; it can contain request headers or credentials.
                    self._record_item_failure(
                        job, item, record.id, "INTERNAL_ERROR", "题目处理失败，请稍后重试", True
                    )
                    terminal = True
                finally:
                    if record.id in job["current_question_ids"]:
                        job["current_question_ids"].remove(record.id)
                    elapsed = (time.perf_counter() - started) * 1000
                    if terminal:
                        job["completed"] += 1
                        item["duration_ms"] = round(elapsed, 1)
                    durations = [
                        value.get("duration_ms")
                        for value in job["items"]
                        if value.get("duration_ms") is not None
                    ]
                    job["average_duration_ms"] = round(sum(durations) / len(durations), 1) if durations else None
                    self._refresh_job_state(job)

        try:
            await asyncio.gather(*(process(record, item) for record, item in zip(selected, job["items"])))
            job["status"] = "cancelled" if job["cancel_requested"] else "completed"
        except asyncio.CancelledError:
            job["cancel_requested"] = True
            job["status"] = "cancelled"
        finally:
            job["finished_at"] = datetime.now(timezone.utc).isoformat()
            self._refresh_job_state(job)
            self._aggregate_metrics(job)
            self.pipeline.runs.add(
                {
                    "run_id": job["job_id"],
                    "job_id": job["job_id"],
                    "type": "batch",
                    "stage": job["stage"],
                    "split": job["split"],
                    "status": job["status"],
                    "duration_ms": self._job_duration(job),
                    "mode": job["mode"],
                    "use_ocr": job["use_ocr"],
                    "ocr_enabled": job["use_ocr"] if job["stage"] == "preliminary" and job["mode"] == "real" else False,
                    "model": self.pipeline.settings.settings.vlm.model or None,
                    "created_at": job["created_at"],
                }
            )

    @staticmethod
    def _new_item(record: Any, position: int) -> dict[str, Any]:
        title, technology = STAGE_DEFAULTS["queued"]
        question_id = getattr(record, "id", record)
        return {
            "id": question_id,
            "position": position,
            "question": str(getattr(record, "question", "") or ""),
            "options": list(getattr(record, "options", None) or []),
            "document": str(getattr(record, "document", "") or ""),
            "page": getattr(record, "page_hint", None),
            "status": "queued",
            "current_stage": "queued",
            "stage_status": "queued",
            "title": title,
            "detail": "等待可用并发槽位",
            "technology": technology,
            "stage_summary": [],
        }

    @staticmethod
    def _set_item_running(item: dict[str, Any]) -> None:
        title, technology = STAGE_DEFAULTS["locating"]
        item.update(
            {
                "status": "running",
                "current_stage": "locating",
                "stage_status": "running",
                "title": title,
                "detail": "正在启动真实分析流水线",
                "technology": technology,
            }
        )

    @staticmethod
    def _update_item_stage(item: dict[str, Any], stage: dict[str, Any]) -> None:
        name = str(stage.get("name") or item.get("current_stage") or "locating")
        status = str(stage.get("status") or "running")
        default_title, default_technology = STAGE_DEFAULTS.get(name, (name, "Analysis Pipeline"))
        public_stage = {
            key: value for key, value in stage.items() if key in PUBLIC_STAGE_KEYS and value is not None
        }
        public_stage.update(
            {
                "name": name,
                "status": status,
                "title": str(stage.get("title") or default_title),
                "detail": str(stage.get("detail") or ""),
                "technology": str(stage.get("technology") or default_technology),
            }
        )
        stages = item["stage_summary"]
        existing_index = next((index for index, value in enumerate(stages) if value["name"] == name), None)
        if existing_index is None:
            stages.append(public_stage)
        else:
            merged = dict(stages[existing_index])
            merged.update(public_stage)
            stages[existing_index] = merged
        item.update(
            {
                "current_stage": name,
                "stage_status": status,
                "title": public_stage["title"],
                "detail": public_stage["detail"],
                "technology": public_stage["technology"],
            }
        )

    @staticmethod
    def _complete_unvisited_stages(item: dict[str, Any], stage: str) -> None:
        visited = {value["name"] for value in item["stage_summary"]}
        for name, _weight in STAGE_PLANS[stage]:
            if name in visited:
                continue
            title, technology = STAGE_DEFAULTS[name]
            item["stage_summary"].append(
                {
                    "name": name,
                    "status": "skipped",
                    "title": title,
                    "detail": "当前执行路径未经过此阶段",
                    "technology": technology,
                }
            )

    @classmethod
    def _record_item_failure(
        cls,
        job: dict[str, Any],
        item: dict[str, Any],
        question_id: str | int,
        code: str,
        message: str,
        retryable: bool,
    ) -> None:
        error = {"code": code, "message": message, "retryable": retryable}
        job["results"].append(
            {"id": question_id, "status": "failed", "answer": "", "error": error}
        )
        item.update({"status": "failed", "stage_status": "failed", "error": error})
        if item.get("stage_summary"):
            active = next(
                (
                    value
                    for value in reversed(item["stage_summary"])
                    if value["name"] == item["current_stage"]
                ),
                item["stage_summary"][-1],
            )
            active["status"] = "failed"
            active["detail"] = message
        else:
            title, technology = STAGE_DEFAULTS.get(
                item["current_stage"], (item["current_stage"], "Analysis Pipeline")
            )
            item["stage_summary"].append(
                {
                    "name": item["current_stage"],
                    "status": "failed",
                    "title": title,
                    "detail": message,
                    "technology": technology,
                }
            )
        job["failed"] += 1

    @classmethod
    def _refresh_job_state(cls, job: dict[str, Any]) -> None:
        items = job["items"]
        running = next((item for item in items if item["status"] == "running"), None)
        if running is not None:
            job["current_stage"] = running["current_stage"]
        elif any(item["status"] == "queued" for item in items):
            job["current_stage"] = "queued"
        elif items:
            job["current_stage"] = items[-1]["current_stage"]

        total = max(1, int(job["total"]))
        job["progress"] = round(sum(cls._item_progress(item, job["stage"]) for item in items) / total, 4)
        job["success_rate"] = round(job["success"] / total, 4)
        error_counts = Counter(
            str(item.get("error", {}).get("code"))
            for item in items
            if item.get("error", {}).get("code")
        )
        job["error_code_counts"] = dict(sorted(error_counts.items()))
        job["stage_summary"] = cls._aggregate_stage_summary(items, job["stage"])
        cls._aggregate_metrics(job)

    @staticmethod
    def _item_progress(item: dict[str, Any], stage: str) -> float:
        if item["status"] in {"completed", "failed"}:
            return 1.0
        status_by_stage = {value["name"]: value["status"] for value in item["stage_summary"]}
        progress = 0.0
        for name, weight in STAGE_PLANS[stage]:
            status = status_by_stage.get(name, "queued")
            if status in {"completed", "skipped", "failed"}:
                progress += weight
            elif status == "running":
                progress += weight * 0.5
        return min(1.0, progress)

    @staticmethod
    def _aggregate_stage_summary(items: list[dict[str, Any]], stage: str) -> dict[str, dict[str, int]]:
        result: dict[str, dict[str, int]] = {}
        for name, _weight in STAGE_PLANS[stage]:
            counts = Counter(
                next(
                    (value["status"] for value in item["stage_summary"] if value["name"] == name),
                    "queued",
                )
                for item in items
            )
            result[name] = {
                "queued": counts["queued"],
                "running": counts["running"],
                "completed": counts["completed"],
                "failed": counts["failed"],
                "skipped": counts["skipped"],
                "total": len(items),
            }
        return result

    @staticmethod
    def _aggregate_metrics(job: dict[str, Any]) -> None:
        metrics = [item.get("result", {}).get("metrics", {}) for item in job["results"]]
        applicable = [metric for metric in metrics if metric.get("applicable")]
        if not applicable:
            job["metrics"] = {
                "applicable": False,
                "accuracy": None,
                "exact_match": None,
                "rouge1_f1": None,
                "semantic_f1": None,
                "bert_f1": None,
                "final_score": None,
            }
            return
        aggregate: dict[str, Any] = {"applicable": True}
        for key in ("accuracy", "exact_match", "rouge1_f1", "semantic_f1", "bert_f1", "final_score"):
            values = [metric.get(key) for metric in applicable if metric.get(key) is not None]
            aggregate[key] = round(sum(values) / len(values), 4) if values else None
        job["metrics"] = aggregate

    async def cancel(self, job_id: str) -> dict[str, Any]:
        job = self._require(job_id)
        if job["status"] in {"completed", "failed", "cancelled"}:
            return self._public(job)
        job["cancel_requested"] = True
        task = self._tasks.get(job_id)
        if task and not task.done():
            task.cancel()
        return self._public(job)

    def get(self, job_id: str) -> dict[str, Any]:
        return self._public(self._require(job_id))

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        values = sorted(self._jobs.values(), key=lambda item: item["created_at"], reverse=True)
        return [self._public(job, include_results=False) for job in values[:limit]]

    def results(self, job_id: str) -> list[dict[str, Any]]:
        return list(self._require(job_id)["results"])

    async def retry_failed(self, job_id: str) -> dict[str, Any]:
        job = self._require(job_id)
        failed_ids = [item["id"] for item in job["results"] if item.get("status") == "failed"]
        if not failed_ids:
            raise AppError("NO_FAILED_ITEMS", "该任务没有可重试的失败项", 409)
        original = job["request"]
        return await self.create(
            JobCreateRequest(
                stage=job["stage"],
                split=job["split"],
                limit=len(failed_ids),
                concurrency=original.get("concurrency", 1),
                question_ids=failed_ids,
                force_demo=original.get("force_demo"),
                use_ocr=original.get("use_ocr", True),
                use_rag=original.get("use_rag", True),
                evaluate_semantic=original.get("evaluate_semantic", False),
            )
        )

    def _require(self, job_id: str) -> dict[str, Any]:
        job = self._jobs.get(job_id)
        if job is None:
            raise AppError("JOB_NOT_FOUND", "未找到指定批量任务", 404)
        return job

    @staticmethod
    def _public(job: dict[str, Any], include_results: bool = True) -> dict[str, Any]:
        result = {key: value for key, value in job.items() if key != "request"}
        result["average_ms"] = result.get("average_duration_ms")
        if not include_results:
            result.pop("results", None)
        return JobManager._sanitize_public(result)

    @staticmethod
    def _sanitize_public(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: JobManager._sanitize_public(item)
                for key, item in value.items()
                if key.lower() not in SECRET_KEYS
            }
        if isinstance(value, list):
            return [JobManager._sanitize_public(item) for item in value]
        if isinstance(value, tuple):
            return [JobManager._sanitize_public(item) for item in value]
        return value

    @staticmethod
    def _job_duration(job: dict[str, Any]) -> float | None:
        if not job.get("started_at") or not job.get("finished_at"):
            return None
        started = datetime.fromisoformat(job["started_at"])
        finished = datetime.fromisoformat(job["finished_at"])
        return round((finished - started).total_seconds() * 1000, 1)
