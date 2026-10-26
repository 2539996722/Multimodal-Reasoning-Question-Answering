import asyncio
import json
from pathlib import Path

from app.config import ModelEndpointConfig
from scripts.precompute_semifinal_ocr import (
    DocumentState,
    PageTask,
    PrecomputePlan,
    atomic_write_text,
    build_plan,
    load_ocr_config,
    run,
)


def _config() -> ModelEndpointConfig:
    return ModelEndpointConfig(
        base_url="https://ocr.example/v1",
        api_key="secret-never-persist",
        model="ocr-model",
        timeout=30,
    )


class _MetadataStub:
    def metadata(self, path):
        del path
        return {"page_count": 3}


def test_load_config_and_manifest_identity_never_expose_api_key(tmp_path) -> None:
    settings = tmp_path / "model-settings.json"
    settings.write_text(
        json.dumps(
            {
                "ocr": {
                    "base_url": "https://ocr.example/v1",
                    "api_key": "secret-never-persist",
                    "model": "ocr-model",
                }
            }
        ),
        encoding="utf-8",
    )
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-test")

    config = load_ocr_config(settings)
    plan = build_plan(
        [("train", pdf)],
        config,
        output_root=tmp_path / "corpus",
        document_service=_MetadataStub(),
    )

    assert config.api_key == "secret-never-persist"
    assert len(plan.tasks) == 3
    assert "secret-never-persist" not in json.dumps(plan.documents[0].manifest)
    assert set(plan.documents[0].manifest["ocr"]) == {"model", "endpoint_hash"}


def test_build_plan_resumes_completed_pages_retries_failed_and_force_rebuilds(tmp_path) -> None:
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-test")
    output = tmp_path / "corpus"
    first = build_plan(
        [("test", pdf)],
        _config(),
        output_root=output,
        document_service=_MetadataStub(),
    )
    state = first.documents[0]
    state.output_dir.mkdir(parents=True)
    (state.output_dir / "page-0001.txt").write_text("done\n", encoding="utf-8")
    state.manifest["pages"] = {
        "1": {"status": "completed", "file": "page-0001.txt", "attempts": 1},
        "2": {"status": "failed", "file": "page-0002.txt", "attempts": 3},
    }
    state.manifest_path.write_text(json.dumps(state.manifest), encoding="utf-8")

    resumed = build_plan(
        [("test", pdf)],
        _config(),
        output_root=output,
        document_service=_MetadataStub(),
    )
    forced = build_plan(
        [("test", pdf)],
        _config(),
        output_root=output,
        document_service=_MetadataStub(),
        force=True,
    )

    assert resumed.skipped_pages == 1
    assert [task.page for task in resumed.tasks] == [2, 3]
    assert [task.page for task in forced.tasks] == [1, 2, 3]


def test_dry_run_never_writes_output(tmp_path) -> None:
    output = tmp_path / "corpus" / "train" / "sample"
    state = DocumentState(
        split="train",
        path=tmp_path / "sample.pdf",
        output_dir=output,
        manifest_path=output / "manifest.json",
        manifest={"pages": {}},
    )
    plan = PrecomputePlan([state], [PageTask(state, 1)], [], 1, 0)

    summary = asyncio.run(run(plan, _config(), concurrency=1, dry_run=True))

    assert summary["planned_pages"] == 1
    assert summary["dry_run"] is True
    assert not output.exists()


def test_atomic_write_replaces_existing_file_without_temporary_artifacts(tmp_path) -> None:
    target = tmp_path / "page-0001.txt"
    target.write_text("old", encoding="utf-8")

    atomic_write_text(target, "new\n")

    assert target.read_text(encoding="utf-8") == "new\n"
    assert list(tmp_path.glob(".*.tmp-*")) == []
