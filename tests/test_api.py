import json
import time
from copy import deepcopy

import pytest
from fastapi.testclient import TestClient

from app.api import routes
from app.main import app


@pytest.fixture()
def client(tmp_path):
    store = routes.settings_store
    original_path = store.persist_path
    original_settings = deepcopy(store.settings)
    store.persist_path = tmp_path / "model-settings.json"
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        store.persist_path = original_path
        store._copy_into(store.settings, original_settings)


def test_health_reports_real_dataset_and_never_performs_model_probe(client: TestClient) -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["dataset_available"] is True
    assert body["dataset_question_count"] == 8402
    assert body["version"]
    assert set(body["models"]) >= {"demo_mode", "effective_mode", "ocr", "vlm", "retrieval", "reranker"}
    assert body["rag"]["framework"] == "LangChain"
    assert body["rag"]["vector_store"].startswith("FAISS")
    assert body["rag"]["vector_index"] == "IndexFlatIP"
    assert body["rag"]["persistence"] == "vectors.npy + documents.json + manifest.json"
    assert "keyword" in body["rag"]["retrieval_strategy"].lower()
    assert body["rag"]["database_required"] is False


def test_health_exposes_liveness_readiness_and_enterprise_headers(client: TestClient) -> None:
    live = client.get("/api/health/live")
    ready = client.get("/api/health/ready")

    assert live.status_code == 200
    assert live.json()["status"] == "ok"
    assert ready.status_code == 200
    assert set(ready.json()["checks"]) == {"dataset", "langchain", "faiss", "runtime_cache"}
    assert set(ready.json()["models_configured"]) == {"ocr", "vlm", "retrieval", "reranker"}
    assert live.headers["x-content-type-options"] == "nosniff"
    assert live.headers["x-frame-options"] == "SAMEORIGIN"
    assert "camera=()" in live.headers["permissions-policy"]
    assert "default-src 'self'" in live.headers["content-security-policy"]
    assert float(live.headers["x-process-time-ms"]) >= 0
    assert live.headers["server-timing"].startswith("app;dur=")


def test_frontend_assets_require_revalidation(client: TestClient) -> None:
    index = client.get("/")
    stylesheet = client.get("/static/css/app.css")

    assert index.status_code == 200
    assert stylesheet.status_code == 200
    assert index.headers["cache-control"] == "no-cache, must-revalidate"
    assert stylesheet.headers["cache-control"] == "no-cache, must-revalidate"


def test_dataset_statistics_match_source_assets(client: TestClient) -> None:
    response = client.get("/api/datasets")

    assert response.status_code == 200
    stages = response.json()["stages"]
    assert stages["preliminary"]["question_count"] == 4300
    assert stages["preliminary"]["document_count"] == 851
    assert stages["semi_final"]["question_count"] == 4102
    assert stages["semi_final"]["document_count"] == 1187


def test_questions_support_pagination_filtering_and_detail(client: TestClient) -> None:
    response = client.get(
        "/api/questions",
        params={
            "stage": "preliminary",
            "split": "test",
            "page": 1,
            "page_size": 2,
            "document": "CN218376896U.pdf",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["page"] == 1
    assert body["page_size"] == 2
    assert body["total"] >= 2
    assert len(body["items"]) == 2
    assert all(item["document"] == "CN218376896U.pdf" for item in body["items"])
    assert all("raw" not in item for item in body["items"])

    detail = client.get("/api/questions/preliminary/test/1")
    assert detail.status_code == 200
    question = detail.json()
    assert question["id"] == 1
    assert question["answer"] == "B"
    assert question["has_reference"] is True


def test_validation_and_missing_resources_use_uniform_error_envelope(client: TestClient) -> None:
    validation = client.get(
        "/api/questions",
        params={"stage": "preliminary", "split": "test", "page_size": 101},
    )
    missing = client.get("/api/questions/preliminary/test/not-an-id")

    assert validation.status_code == 422
    assert missing.status_code == 404
    for response in (validation, missing):
        error = response.json()["error"]
        assert set(error) >= {"code", "message", "request_id", "retryable"}
        assert error["request_id"]


@pytest.mark.parametrize(
    "encoded_filename",
    [
        "questions.jsonl",
        "%2E%2E%5Cquestions.jsonl",
        "C%3A%5CWindows%5Cwin.ini",
        "unknown.pdf",
    ],
)
def test_document_api_rejects_non_pdf_traversal_and_unknown_files(
    client: TestClient, encoded_filename: str
) -> None:
    response = client.get(
        f"/api/documents/preliminary/train/{encoded_filename}",
        follow_redirects=False,
    )

    assert response.status_code in {400, 404}
    assert "根据专利文本" not in response.text
    error = response.json()["error"]
    assert error["code"] in {"INVALID_DOCUMENT_PATH", "DOCUMENT_NOT_FOUND", "NOT_FOUND"}


def test_allowlisted_pdf_is_previewable_with_security_headers(client: TestClient) -> None:
    response = client.get("/api/documents/preliminary/train/CN213444549U.pdf")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/pdf")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.content.startswith(b"%PDF")


def test_document_ocr_returns_page_items_for_cached_and_missing_pages(
    client: TestClient, monkeypatch
) -> None:
    monkeypatch.setattr(
        routes.rag_service.index,
        "load_txt_ocr_corpus",
        lambda _path, _model, **_kwargs: {
            "pages": {1: "第一页 OCR", 3: "第三页 OCR"},
            "meta": {"source": "precomputed_corpus", "stale": False, "stale_reason": None},
        },
    )
    monkeypatch.setattr(routes.document_service, "metadata", lambda _path: {"page_count": 3})

    response = client.get("/api/documents/preliminary/train/CN213444549U.pdf/ocr")

    assert response.status_code == 200
    body = response.json()
    assert body["pages"] == [1, 3]
    assert body["page_count"] == 2
    assert body["total_pages"] == 3
    assert body["page_items"] == [
        {"page": 1, "cached": True, "characters": 7, "text": "第一页 OCR"},
        {"page": 2, "cached": False, "characters": 0, "text": ""},
        {"page": 3, "cached": True, "characters": 7, "text": "第三页 OCR"},
    ]


def test_api_keys_are_write_only_across_public_responses(client: TestClient) -> None:
    secret = "sk-api-contract-never-return-this"
    try:
        updated = client.put(
            "/api/settings/models",
            json={
                "demo_mode": True,
                "ocr": {
                    "base_url": "https://ocr.example/v1",
                    "model": "ocr-test",
                    "api_key": secret,
                },
                "vlm": {
                    "base_url": "https://vlm.example/v1",
                    "model": "vlm-test",
                    "api_key": secret,
                },
                "retrieval": {
                    "base_url": "https://retrieval.example/v1",
                    "model": "embedding-test",
                    "api_key": secret,
                },
                "reranker": {
                    "base_url": "https://reranker.example/v1",
                    "model": "reranker-test",
                    "api_key": secret,
                },
            },
        )
        settings = client.get("/api/settings/models")
        health = client.get("/api/health")
        runs = client.get("/api/runs")

        assert updated.status_code == 200
        assert settings.status_code == 200
        public_payload = json.dumps(
            [updated.json(), settings.json(), health.json(), runs.json()], ensure_ascii=False
        )
        assert secret not in public_payload
        assert set(settings.json()) >= {"ocr", "vlm", "retrieval", "reranker"}
        for kind in ("ocr", "vlm", "retrieval", "reranker"):
            assert settings.json()[kind]["api_key_set"] is True
            assert "api_key" not in settings.json()[kind]
    finally:
        client.put(
            "/api/settings/models",
            json={
                "demo_mode": True,
                "ocr": {"clear_api_key": True},
                "vlm": {"clear_api_key": True},
                "retrieval": {"clear_api_key": True},
                "reranker": {"clear_api_key": True},
            },
        )


def test_demo_analyze_replays_reference_without_claiming_real_model(client: TestClient) -> None:
    response = client.post(
        "/api/analyze",
        json={
            "stage": "preliminary",
            "split": "train",
            "question_id": 1,
            "force_demo": True,
        },
    )

    assert response.status_code == 200
    body = response.json()
    result = body["result"]
    assert body["status"] == "completed"
    assert result["answer"] == "B"
    assert result["mode"] == "demo"
    assert result["demo_reference_replay"] is True
    assert result["metrics"]["accuracy"] == 1.0
    assert result["models"] == {"ocr": None, "vlm": None}
    assert "演示模式" in result["ocr_text"]
    assert result["execution"]["engine"] == "Reference Replay Engine"
    assert result["reasoning_summary"]
    trace = result["technical_trace"]
    assert [item["name"] for item in trace] == ["locating", "ocr", "reasoning", "validation"]
    assert next(item for item in trace if item["name"] == "ocr")["status"] == "skipped"
    assert all(item.get("technology") for item in trace)


def test_demo_analyze_stream_emits_real_stage_events_and_final_result(client: TestClient) -> None:
    with client.stream(
        "POST",
        "/api/analyze/stream",
        json={
            "stage": "preliminary",
            "split": "train",
            "question_id": 1,
            "force_demo": True,
        },
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        lines = [line for line in response.iter_lines() if line]

    event_names = [line.removeprefix("event: ") for line in lines if line.startswith("event: ")]
    data_rows = [json.loads(line.removeprefix("data: ")) for line in lines if line.startswith("data: ")]
    assert event_names.count("stage") >= 4
    assert event_names[-1] == "result"
    assert data_rows[-1]["result"]["answer"] == "B"
    assert data_rows[-1]["result"]["mode"] == "demo"


def test_batch_job_completes_and_exports_strict_jsonl(
    client: TestClient, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(routes.export_service, "export_dir", tmp_path)
    created = client.post(
        "/api/jobs",
        json={
            "stage": "preliminary",
            "split": "train",
            "question_ids": [1, 2],
            "limit": 2,
            "concurrency": 2,
            "force_demo": True,
        },
    )

    assert created.status_code == 202
    job_id = created.json()["job_id"]
    body = None
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        polled = client.get(f"/api/jobs/{job_id}")
        assert polled.status_code == 200
        body = polled.json()
        if body["status"] in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.02)

    assert body is not None
    assert body["status"] == "completed"
    assert body["total"] == 2
    assert body["completed"] == 2
    assert body["success"] == 2
    assert body["failed"] == 0
    assert body["progress"] == 1.0
    assert body["metrics"]["accuracy"] == 1.0

    exported = client.get(f"/api/jobs/{job_id}/export")
    assert exported.status_code == 200
    assert exported.headers["content-type"].startswith("application/x-ndjson")
    rows = [json.loads(line) for line in exported.text.splitlines()]
    assert len(rows) == 2
    assert all(set(row) == {"id", "answer"} for row in rows)
    assert all(row["answer"] in {"A", "B", "C", "D"} for row in rows)
