from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from app.config import ROOT_DIR
from app.errors import AppError
from app.models.schemas import QuestionRecord, Split, Stage


PAGE_PATTERN = re.compile(r"第\s*(\d{1,4})\s*页")
STAGE_LABELS = {"preliminary": "初赛", "semi_final": "复赛"}
TASK_TYPES = {"preliminary": "multiple_choice", "semi_final": "open_ended"}


@dataclass(slots=True)
class PartitionIndex:
    stage: Stage
    split: Split
    source_files: list[str] = field(default_factory=list)
    records: list[QuestionRecord] = field(default_factory=list)
    by_id: dict[str, QuestionRecord] = field(default_factory=dict)
    documents: dict[str, Path] = field(default_factory=dict)
    anomalies: list[dict[str, Any]] = field(default_factory=list)


class DatasetRepository:
    """Read-only in-memory index over the competition JSONL and PDF assets."""

    STAGE_DIRS: dict[Stage, str] = {
        "preliminary": "preliminary_dataset",
        "semi_final": "semi_final_dataset",
    }
    QUESTION_FILES: dict[tuple[Stage, Split], tuple[str, ...]] = {
        ("preliminary", "train"): ("questions.jsonl",),
        ("preliminary", "test"): ("questions.jsonl",),
        ("semi_final", "train"): ("train.jsonl",),
        ("semi_final", "test"): ("test.jsonl",),
    }

    def __init__(self, root_dir: Path | str = ROOT_DIR) -> None:
        self.root_dir = Path(root_dir).resolve()
        self._partitions: dict[tuple[Stage, Split], PartitionIndex] = {}
        self.indexed_at = datetime.now(timezone.utc)
        self.scan()

    def scan(self) -> None:
        partitions: dict[tuple[Stage, Split], PartitionIndex] = {}
        for stage in self.STAGE_DIRS:
            for split in ("train", "test"):
                partitions[(stage, split)] = self._scan_partition(stage, split)
        self._partitions = partitions
        self.indexed_at = datetime.now(timezone.utc)

    def _scan_partition(self, stage: Stage, split: Split) -> PartitionIndex:
        partition = PartitionIndex(stage=stage, split=split)
        split_dir = (self.root_dir / self.STAGE_DIRS[stage] / split).resolve()
        if not split_dir.is_dir():
            partition.anomalies.append({"code": "SPLIT_MISSING", "message": f"数据分区不存在: {split_dir.name}"})
            return partition

        documents_dir = (split_dir / "documents").resolve()
        if documents_dir.is_dir():
            for path in documents_dir.iterdir():
                if path.is_file() and path.suffix.lower() == ".pdf":
                    partition.documents[path.name.casefold()] = path.resolve()
        else:
            partition.anomalies.append({"code": "DOCUMENTS_DIR_MISSING", "message": "documents 目录不存在"})

        raw_records: list[dict[str, Any]] = []
        for filename in self.QUESTION_FILES[(stage, split)]:
            path = split_dir / filename
            if path.is_file():
                partition.source_files.append(filename)
                raw_records.extend(self._read_jsonl(path, partition.anomalies))
            else:
                partition.anomalies.append({"code": "QUESTION_FILE_MISSING", "message": f"缺少 {filename}"})

        answers: dict[str, str] = {}
        answer_path = split_dir / "answers.jsonl"
        if answer_path.is_file():
            partition.source_files.append(answer_path.name)
            for row in self._read_jsonl(answer_path, partition.anomalies):
                if "id" in row and row.get("answer") is not None:
                    answers[str(row["id"])] = str(row["answer"])

        id_counts = Counter(str(row.get("id")) for row in raw_records if row.get("id") is not None)
        for duplicate_id, count in id_counts.items():
            if count > 1:
                partition.anomalies.append(
                    {"code": "DUPLICATE_ID", "message": f"题目 ID {duplicate_id} 重复 {count} 次"}
                )

        for row_number, raw in enumerate(raw_records, start=1):
            if raw.get("id") is None or not raw.get("question") or not raw.get("document"):
                partition.anomalies.append(
                    {"code": "INVALID_QUESTION", "message": f"第 {row_number} 条记录缺少必要字段"}
                )
                continue
            qid = raw["id"]
            answer_value = raw.get("answer", answers.get(str(qid)))
            document = str(raw["document"])
            options = raw.get("options")
            if options is not None and not isinstance(options, list):
                options = [str(options)]
            record = QuestionRecord(
                stage=stage,
                split=split,
                id=qid,
                question=str(raw["question"]),
                document=document,
                options=[str(value) for value in options] if options else None,
                answer=str(answer_value) if answer_value is not None else None,
                group=raw.get("group"),
                page_hint=self.extract_page_hint(str(raw["question"])),
                has_reference=answer_value is not None,
                document_available=document.casefold() in partition.documents,
                raw=raw,
            )
            partition.records.append(record)
            partition.by_id.setdefault(str(qid), record)
            if not record.document_available:
                partition.anomalies.append(
                    {"code": "DOCUMENT_MISSING", "message": f"题目 {qid} 引用的文档不存在", "document": document}
                )
        return partition

    @staticmethod
    def _read_jsonl(path: Path, anomalies: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError("JSON value is not an object")
                    rows.append(value)
                except (json.JSONDecodeError, ValueError) as exc:
                    anomalies.append(
                        {"code": "INVALID_JSONL", "message": f"{path.name}:{line_number}: {exc}"}
                    )
        return rows

    @staticmethod
    def extract_page_hint(question: str) -> int | None:
        match = PAGE_PATTERN.search(question)
        if not match:
            return None
        page = int(match.group(1))
        return page if page >= 1 else None

    def partition(self, stage: Stage, split: Split) -> PartitionIndex:
        try:
            return self._partitions[(stage, split)]
        except KeyError as exc:
            raise AppError("DATASET_NOT_FOUND", "未找到指定数据集分区", 404) from exc

    def datasets(self) -> dict[str, Any]:
        stages: list[dict[str, Any]] = []
        keyed: dict[str, Any] = {}
        for stage in self.STAGE_DIRS:
            split_items: list[dict[str, Any]] = []
            for split in ("train", "test"):
                part = self.partition(stage, split)
                split_items.append(
                    {
                        "id": split,
                        "question_count": len(part.records),
                        "document_count": len(part.documents),
                        "reference_count": sum(record.has_reference for record in part.records),
                        "has_references": any(record.has_reference for record in part.records),
                        "source_files": part.source_files,
                        "anomalies": part.anomalies[:100],
                        "anomaly_count": len(part.anomalies),
                    }
                )
            item = {
                "id": stage,
                "stage": stage,
                "name": STAGE_LABELS[stage],
                "task_type": TASK_TYPES[stage],
                "question_count": sum(value["question_count"] for value in split_items),
                "document_count": sum(value["document_count"] for value in split_items),
                "splits": split_items,
                "train": split_items[0],
                "test": split_items[1],
                "anomalies": [entry for value in split_items for entry in value["anomalies"]],
            }
            stages.append(item)
            keyed[stage] = item
        return {"items": stages, "stages": keyed, "indexed_at": self.indexed_at.isoformat()}

    def query_questions(
        self,
        stage: Stage,
        split: Split,
        page: int = 1,
        page_size: int = 30,
        keyword: str | None = None,
        document: str | None = None,
        question_id: str | None = None,
    ) -> dict[str, Any]:
        records: Iterable[QuestionRecord] = self.partition(stage, split).records
        if question_id is not None:
            records = (record for record in records if str(record.id) == str(question_id))
        if document:
            document_term = document.casefold()
            records = (record for record in records if document_term in record.document.casefold())
        if keyword:
            term = keyword.casefold()
            records = (
                record
                for record in records
                if term in record.question.casefold()
                or term in record.document.casefold()
                or term in str(record.id).casefold()
            )
        filtered = list(records)
        total = len(filtered)
        start = (page - 1) * page_size
        items = [record.model_dump() for record in filtered[start : start + page_size]]
        return {
            "items": items,
            "page": page,
            "page_size": page_size,
            "total": total,
            "pages": (total + page_size - 1) // page_size if total else 0,
        }

    def get_question(self, stage: Stage, split: Split, question_id: str | int) -> QuestionRecord:
        record = self.partition(stage, split).by_id.get(str(question_id))
        if record is None:
            raise AppError("QUESTION_NOT_FOUND", "未找到指定题目", 404)
        return record

    def resolve_document(self, stage: Stage, split: Split, filename: str) -> Path:
        if not filename or filename != Path(filename).name or Path(filename).suffix.lower() != ".pdf":
            raise AppError("INVALID_DOCUMENT_PATH", "文档路径不合法", 400)
        partition = self.partition(stage, split)
        path = partition.documents.get(filename.casefold())
        if path is None:
            raise AppError("DOCUMENT_NOT_FOUND", "未找到指定文档", 404)
        documents_dir = (self.root_dir / self.STAGE_DIRS[stage] / split / "documents").resolve()
        try:
            path.relative_to(documents_dir)
        except ValueError as exc:
            raise AppError("INVALID_DOCUMENT_PATH", "文档路径超出允许目录", 400) from exc
        return path
