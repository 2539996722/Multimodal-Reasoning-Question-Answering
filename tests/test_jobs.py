from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient

from app.api import routes
from app.config import AppSettings
from app.errors import AppError
from app.main import app
from app.models.schemas import JobCreateRequest
from app.services.jobs import JobManager


class FakeRepository:
    def __init__(self, ids: tuple[int, ...] = (1, 2)) -> None:
        self.records = [SimpleNamespace(id=value, question=f"测试问题 {value}", options=["A. 选项", "B. 选项"], document="demo.pdf", page_hint=2) for value in ids]

    def partition(self, _stage: str, _split: str) -> Any:
        return SimpleNamespace(records=self.records)

    def get_question(self, _stage: str, _split: str, question_id: int) -> Any:
        return next(record for record in self.records if record.id == question_id)


class FakeRuns:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def add(self, item: dict[str, Any]) -> None:
        self.items.append(item)


class ControlledPipeline:
    def __init__(self) -> None:
        self.runs = FakeRuns()
        self.settings = SimpleNamespace(settings=SimpleNamespace(vlm=SimpleNamespace(model="test-vlm")))
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.force_demo_values: list[bool | None] = []
        self.fail_ids: set[int] = set()

    async def analyze(self, request, source="single", event_callback=None):
        assert source == "batch_item"
        self.force_demo_values.append(request.force_demo)
        event_callback(
            {
                "name": "locating",
                "status": "completed",
                "title": "文档定位",
                "detail": "定位完成",
                "technology": "PyMuPDF",
                "api_key": "must-not-leak",
            }
        )
        event_callback(
            {
                "name": "ocr",
                "status": "running",
                "title": "OCR 文档识别",
                "detail": "正在识别",
                "technology": "OpenAI-compatible OCR",
                "authorization": "Bearer must-not-leak",
            }
        )
        self.entered.set()
        await self.release.wait()
        if request.question_id in self.fail_ids:
            raise AppError("MODEL_TIMEOUT", "模型请求超时", 504, retryable=True)
        event_callback(
            {
                "name": "ocr",
                "status": "completed",
                "title": "OCR 文档识别",
                "detail": "识别完成",
                "technology": "OpenAI-compatible OCR",
            }
        )
        for name in ("reasoning", "validation"):
            event_callback(
                {
                    "name": name,
                    "status": "completed",
                    "title": name,
                    "detail": "完成",
                    "technology": "test pipeline",
                }
            )
        return {
            "run_id": f"run-{request.question_id}",
            "result": {
                "answer": "A",
                "answer_explanation": "依据文档证据选择 A。",
                "reasoning_summary": ["读取 OCR 上下文。", "核对目标页后选择 A。"],
                "evidence": [{"page": 2, "source": "ocr", "quote": "关键证据"}],
                "option_analysis": [{"option": "A", "verdict": "selected", "reason": "证据匹配"}],
                "duration_ms": 12.0,
                "metrics": {
                    "applicable": True,
                    "accuracy": 1.0,
                    "exact_match": None,
                    "rouge1_f1": None,
                },
            },
        }


def test_batch_defaults_to_real_and_exposes_weighted_stage_progress_without_secrets() -> None:
    async def scenario() -> None:
        pipeline = ControlledPipeline()
        manager = JobManager(FakeRepository((1,)), pipeline, AppSettings(demo_mode=True))
        created = await manager.create(
            JobCreateRequest(stage="preliminary", split="train", limit=1)
        )
        job_id = created["job_id"]
        await pipeline.entered.wait()

        running = manager.get(job_id)
        assert running["mode"] == "real"
        assert pipeline.force_demo_values == [False]
        assert running["completed"] == 0
        assert 0 < running["progress"] < 1
        assert running["current_stage"] == "ocr"
        assert running["items"][0]["status"] == "running"
        assert running["items"][0]["stage_status"] == "running"
        assert running["items"][0]["title"] == "OCR 文档识别"
        assert running["items"][0]["question"] == "测试问题 1"
        assert running["stage_summary"]["ocr"]["running"] == 1
        assert "must-not-leak" not in str(running)
        assert "api_key" not in str(running)
        assert "authorization" not in str(running)

        pipeline.release.set()
        await manager._tasks[job_id]
        completed = manager.get(job_id)
        assert completed["status"] == "completed"
        assert completed["progress"] == 1.0
        assert completed["completed"] == 1
        assert completed["success_rate"] == 1.0
        assert completed["error_code_counts"] == {}
        assert completed["items"][0]["answer"] == "A"
        assert completed["items"][0]["reasoning_summary"][0] == "读取 OCR 上下文。"
        assert completed["items"][0]["evidence"][0]["page"] == 2

    asyncio.run(scenario())


def test_failed_item_is_retryable_and_retry_preserves_normalized_real_mode() -> None:
    async def scenario() -> None:
        pipeline = ControlledPipeline()
        pipeline.fail_ids.add(2)
        manager = JobManager(FakeRepository((1, 2)), pipeline, AppSettings(demo_mode=True))
        created = await manager.create(
            JobCreateRequest(
                stage="preliminary",
                split="train",
                question_ids=[1, 2],
                limit=2,
                concurrency=2,
            )
        )
        job_id = created["job_id"]
        await pipeline.entered.wait()
        pipeline.release.set()
        await manager._tasks[job_id]

        completed = manager.get(job_id)
        assert completed["completed"] == 2
        assert completed["success"] == 1
        assert completed["failed"] == 1
        assert completed["success_rate"] == 0.5
        assert completed["error_code_counts"] == {"MODEL_TIMEOUT": 1}
        assert [item["status"] for item in completed["items"]] == ["completed", "failed"]

        pipeline.fail_ids.clear()
        retried = await manager.retry_failed(job_id)
        retry_id = retried["job_id"]
        await manager._tasks[retry_id]
        retry_result = manager.get(retry_id)
        assert retry_result["mode"] == "real"
        assert retry_result["total"] == 1
        assert retry_result["success"] == 1
        assert all(value is False for value in pipeline.force_demo_values)

    asyncio.run(scenario())


def test_explicit_force_demo_is_the_only_demo_opt_in() -> None:
    async def scenario() -> None:
        pipeline = ControlledPipeline()
        pipeline.release.set()
        manager = JobManager(FakeRepository((1,)), pipeline, AppSettings(demo_mode=False))
        created = await manager.create(
            JobCreateRequest(
                stage="preliminary", split="train", limit=1, force_demo=True
            )
        )
        await manager._tasks[created["job_id"]]
        assert manager.get(created["job_id"])["mode"] == "demo"
        assert pipeline.force_demo_values == [True]

    asyncio.run(scenario())


def test_jobs_api_omitted_force_demo_runs_real_and_returns_item_stage_contract(monkeypatch) -> None:
    observed: list[bool | None] = []

    async def fake_analyze(request, source="single", event_callback=None):
        observed.append(request.force_demo)
        for name in ("locating", "ocr", "reasoning", "validation"):
            event_callback(
                {
                    "name": name,
                    "status": "completed",
                    "title": f"{name} title",
                    "detail": f"{name} detail",
                    "technology": f"{name} technology",
                }
            )
        return {
            "run_id": "api-test-run",
            "result": {
                "answer": "B",
                "duration_ms": 8.0,
                "metrics": {
                    "applicable": True,
                    "accuracy": 1.0,
                    "exact_match": None,
                    "rouge1_f1": None,
                },
            },
        }

    monkeypatch.setattr(routes.analysis_pipeline, "analyze", fake_analyze)
    with TestClient(app) as client:
        response = client.post(
            "/api/jobs",
            json={
                "stage": "preliminary",
                "split": "train",
                "question_ids": [1],
                "limit": 1,
            },
        )
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        body = None
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            body = client.get(f"/api/jobs/{job_id}").json()
            if body["status"] == "completed":
                break
            time.sleep(0.01)

    assert body is not None
    assert body["status"] == "completed"
    assert body["mode"] == "real"
    assert observed == [False]
    assert body["progress"] == 1.0
    assert body["current_stage"] == "validation"
    assert body["items"][0]["status"] == "completed"
    assert body["items"][0]["stage_status"] == "completed"
    assert [value["name"] for value in body["items"][0]["stage_summary"]] == [
        "locating",
        "ocr",
        "reasoning",
        "validation",
    ]
    assert set(body["stage_summary"]) == {"locating", "ocr", "reasoning", "validation"}


def test_cancel_keeps_a_retryable_terminal_record_for_the_interrupted_item() -> None:
    async def scenario() -> None:
        pipeline = ControlledPipeline()
        manager = JobManager(FakeRepository((1, 2)), pipeline, AppSettings())
        created = await manager.create(
            JobCreateRequest(
                stage="preliminary", split="train", limit=2, concurrency=1
            )
        )
        job_id = created["job_id"]
        await pipeline.entered.wait()
        response = await manager.cancel(job_id)
        assert response["cancel_requested"] is True
        await manager._tasks[job_id]

        cancelled = manager.get(job_id)
        assert cancelled["status"] == "cancelled"
        assert cancelled["completed"] == 1
        assert cancelled["failed"] == 1
        assert cancelled["error_code_counts"] == {"JOB_CANCELLED": 1}
        assert [item["status"] for item in cancelled["items"]] == ["failed", "queued"]
        assert manager.results(job_id)[0]["error"]["retryable"] is True

    asyncio.run(scenario())
