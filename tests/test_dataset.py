import json
from pathlib import Path

import pytest

from app.errors import AppError
from app.services.dataset import DatasetRepository


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


@pytest.fixture()
def dataset_root(tmp_path: Path) -> Path:
    partitions = {
        ("preliminary_dataset", "train"): (
            "questions.jsonl",
            [
                {
                    "id": 1,
                    "question": "查看第 10 页，选择正确部件",
                    "document": "PRE-TRAIN.pdf",
                    "options": ["A. 左", "B. 右", "C. 上", "D. 下"],
                    "answer": "B",
                }
            ],
        ),
        ("preliminary_dataset", "test"): (
            "questions.jsonl",
            [
                {
                    "id": 2,
                    "question": "测试题",
                    "document": "pre-test.PDF",
                    "options": ["A", "B", "C", "D"],
                }
            ],
        ),
        ("semi_final_dataset", "train"): (
            "train.jsonl",
            [
                {
                    "id": "open-1",
                    "question": "装置作用是什么？",
                    "document": "semi-train.pdf",
                    "answer": "用于定位",
                }
            ],
        ),
        ("semi_final_dataset", "test"): (
            "test.jsonl",
            [{"id": 4, "question": "连接关系是什么？", "document": "semi-test.pdf"}],
        ),
    }

    for (stage_dir, split), (filename, rows) in partitions.items():
        split_dir = tmp_path / stage_dir / split
        documents = split_dir / "documents"
        documents.mkdir(parents=True)
        document_name = rows[0]["document"]
        (documents / document_name).write_bytes(b"%PDF-1.4\n%%EOF\n")
        write_jsonl(split_dir / filename, rows)

    write_jsonl(
        tmp_path / "preliminary_dataset" / "test" / "answers.jsonl",
        [{"id": 2, "answer": "C"}],
    )
    write_jsonl(
        tmp_path / "semi_final_dataset" / "test" / "answers.jsonl",
        [{"id": 4, "answer": "直接连接"}],
    )
    return tmp_path


def test_discovers_all_partitions_and_merges_separate_answers(dataset_root: Path) -> None:
    repository = DatasetRepository(dataset_root)

    summary = repository.datasets()
    preliminary = summary["stages"]["preliminary"]
    semi_final = summary["stages"]["semi_final"]

    assert preliminary["question_count"] == 2
    assert preliminary["document_count"] == 2
    assert semi_final["question_count"] == 2
    assert semi_final["document_count"] == 2

    preliminary_test = repository.get_question("preliminary", "test", 2)
    semi_final_test = repository.get_question("semi_final", "test", 4)
    assert preliminary_test.answer == "C"
    assert preliminary_test.has_reference is True
    assert semi_final_test.answer == "直接连接"
    assert semi_final_test.has_reference is True


def test_query_is_paginated_searchable_and_extracts_page_hint(dataset_root: Path) -> None:
    repository = DatasetRepository(dataset_root)

    result = repository.query_questions(
        "preliminary", "train", page=1, page_size=1, keyword="部件"
    )

    assert result["total"] == 1
    assert result["pages"] == 1
    assert len(result["items"]) == 1
    assert result["items"][0]["page_hint"] == 10
    assert "raw" not in result["items"][0]


@pytest.mark.parametrize(
    "filename",
    [
        "../questions.jsonl",
        "..\\questions.jsonl",
        "C:\\Windows\\win.ini",
        "notes.txt",
        "missing.pdf",
    ],
)
def test_document_resolution_rejects_traversal_and_non_allowlisted_files(
    dataset_root: Path, filename: str
) -> None:
    repository = DatasetRepository(dataset_root)

    with pytest.raises(AppError) as caught:
        repository.resolve_document("preliminary", "train", filename)

    assert caught.value.code in {"INVALID_DOCUMENT_PATH", "DOCUMENT_NOT_FOUND"}


def test_document_resolution_is_case_insensitive_but_stays_in_allowlist(dataset_root: Path) -> None:
    repository = DatasetRepository(dataset_root)

    resolved = repository.resolve_document("preliminary", "train", "pre-train.PDF")

    assert resolved.name == "PRE-TRAIN.pdf"
    assert resolved.parent == (
        dataset_root / "preliminary_dataset" / "train" / "documents"
    ).resolve()


def test_bad_jsonl_and_duplicate_ids_are_reported_without_crashing(tmp_path: Path) -> None:
    split_dir = tmp_path / "preliminary_dataset" / "train"
    (split_dir / "documents").mkdir(parents=True)
    (split_dir / "documents" / "a.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
    (split_dir / "questions.jsonl").write_text(
        '{"id": 1, "question": "q1", "document": "a.pdf"}\n'
        "this-is-not-json\n"
        '{"id": 1, "question": "q2", "document": "a.pdf"}\n',
        encoding="utf-8",
    )

    repository = DatasetRepository(tmp_path)
    partition = repository.partition("preliminary", "train")
    codes = {item["code"] for item in partition.anomalies}

    assert len(partition.records) == 2
    assert "INVALID_JSONL" in codes
    assert "DUPLICATE_ID" in codes

