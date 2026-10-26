import json
from pathlib import Path

from app.services.export import ExportService


def test_export_is_utf8_jsonl_with_only_competition_fields() -> None:
    payload = ExportService.render_jsonl(
        [
            {"id": 1, "answer": "B", "reasoning": "must not leak"},
            {"id": "复-2", "result": {"answer": "用于定位", "tokens": 42}},
        ]
    )

    text = payload.decode("utf-8")
    rows = [json.loads(line) for line in text.splitlines()]
    assert text.endswith("\n")
    assert rows == [
        {"id": 1, "answer": "B"},
        {"id": "复-2", "answer": "用于定位"},
    ]
    assert all(set(row) == {"id", "answer"} for row in rows)


def test_export_files_are_not_overwritten(tmp_path: Path) -> None:
    service = ExportService(tmp_path)

    first = service.write_jsonl("preliminary", [{"id": 1, "answer": "A"}])
    second = service.write_jsonl("preliminary", [{"id": 1, "answer": "B"}])

    assert first.exists()
    assert second.exists()
    assert first != second
    assert json.loads(first.read_text(encoding="utf-8")) == {"id": 1, "answer": "A"}

