from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import ModelEndpointConfig  # noqa: E402
from app.errors import AppError  # noqa: E402
from app.services.documents import DocumentService  # noqa: E402
from app.services.openai_client import OpenAICompatibleClient, chat_completions_url  # noqa: E402


SETTINGS_PATH = ROOT_DIR / "runtime" / "model-settings.json"
DATASET_ROOT = ROOT_DIR / "semi_final_dataset"
OUTPUT_ROOT = ROOT_DIR / "runtime" / "ocr-corpus"
INTER_PAGE_DELAY_SECONDS = 1.0
OCR_SYSTEM_PROMPT = (
    "你是工业技术文档 OCR 引擎。忠实识别页面所有可见内容，保留段落、部件编号、表格行列、"
    "公式、流程箭头和图中标注。无法辨认处标记为[无法辨认]，不得编造。只输出识别文本。"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        temporary.write_text(content, encoding="utf-8")
        last_error: PermissionError | None = None
        for attempt in range(6):
            try:
                os.replace(temporary, path)
                last_error = None
                break
            except PermissionError as exc:
                last_error = exc
                if attempt < 5:
                    time.sleep(0.25 * (attempt + 1))
        if last_error is not None:
            raise last_error
    finally:
        temporary.unlink(missing_ok=True)


def load_ocr_config(settings_path: Path = SETTINGS_PATH) -> ModelEndpointConfig:
    try:
        payload = json.loads(settings_path.read_text(encoding="utf-8"))
        value = payload["ocr"]
    except FileNotFoundError as exc:
        raise RuntimeError(f"模型配置文件不存在: {settings_path}") from exc
    except (ValueError, KeyError, TypeError) as exc:
        raise RuntimeError("模型配置文件缺少合法的 ocr 配置") from exc
    config = ModelEndpointConfig(
        base_url=str(value.get("base_url") or ""),
        api_key=str(value.get("api_key") or ""),
        model=str(value.get("model") or ""),
        timeout=float(value.get("timeout") or 120),
        temperature=float(value.get("temperature") or 0),
    )
    if not config.configured:
        raise RuntimeError("OCR 配置不完整，请先在 runtime/model-settings.json 配置 base_url、api_key 和 model")
    return config


def discover_documents(
    dataset_root: Path = DATASET_ROOT,
    splits: Iterable[str] = ("train", "test"),
    selectors: Iterable[str] = (),
) -> list[tuple[str, Path]]:
    selected = {value.strip().casefold() for raw in selectors for value in raw.split(",") if value.strip()}
    documents: list[tuple[str, Path]] = []
    for split in splits:
        directory = dataset_root / split / "documents"
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.pdf"), key=lambda item: item.name.casefold()):
            if selected and path.name.casefold() not in selected and path.stem.casefold() not in selected:
                continue
            documents.append((split, path.resolve()))
    if selected:
        found = {path.name.casefold() for _, path in documents} | {path.stem.casefold() for _, path in documents}
        missing = sorted(value for value in selected if value not in found)
        if missing:
            raise RuntimeError(f"未找到指定复赛 PDF: {', '.join(missing)}")
    return documents


@dataclass(slots=True)
class DocumentState:
    split: str
    path: Path
    output_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass(slots=True)
class PageTask:
    document: DocumentState
    page: int


@dataclass(slots=True)
class PrecomputePlan:
    documents: list[DocumentState]
    tasks: list[PageTask]
    planning_errors: list[dict[str, str]]
    total_pages: int
    skipped_pages: int


def _source_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    seed = f"{path.name}|{stat.st_size}|{stat.st_mtime_ns}"
    return {
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "fingerprint": hashlib.sha256(seed.encode("utf-8")).hexdigest(),
    }


def _config_identity(config: ModelEndpointConfig) -> dict[str, Any]:
    endpoint = chat_completions_url(config.base_url)
    return {
        "model": config.model,
        "endpoint_hash": hashlib.sha256(endpoint.encode("utf-8")).hexdigest()[:16],
    }


def _new_manifest(
    split: str,
    path: Path,
    page_count: int,
    config: ModelEndpointConfig,
    render_scale: float,
) -> dict[str, Any]:
    now = utc_now()
    return {
        "schema_version": 1,
        "split": split,
        "document": path.name,
        "source": _source_fingerprint(path),
        "ocr": _config_identity(config),
        "render_scale": render_scale,
        "page_count": page_count,
        "created_at": now,
        "updated_at": now,
        "completed_pages": 0,
        "failed_pages": 0,
        "pages": {},
    }


def _load_manifest(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def _manifest_matches(existing: dict[str, Any], expected: dict[str, Any]) -> bool:
    return all(
        existing.get(key) == expected.get(key)
        for key in ("schema_version", "split", "document", "source", "ocr", "render_scale", "page_count")
    )


def _refresh_counts(manifest: dict[str, Any]) -> None:
    pages = manifest.get("pages", {})
    manifest["completed_pages"] = sum(value.get("status") == "completed" for value in pages.values())
    manifest["failed_pages"] = sum(value.get("status") == "failed" for value in pages.values())
    manifest["updated_at"] = utc_now()


def build_plan(
    documents: list[tuple[str, Path]],
    config: ModelEndpointConfig,
    *,
    output_root: Path = OUTPUT_ROOT,
    document_service: DocumentService | None = None,
    force: bool = False,
    max_pages: int | None = None,
    render_scale: float = 1.8,
) -> PrecomputePlan:
    service = document_service or DocumentService()
    states: list[DocumentState] = []
    tasks: list[PageTask] = []
    errors: list[dict[str, str]] = []
    total_pages = 0
    skipped_pages = 0
    for split, path in documents:
        try:
            page_count = int(service.metadata(path).get("page_count") or 0)
            if page_count < 1:
                raise RuntimeError("PDF 页数为 0 或无法读取")
        except Exception as exc:
            errors.append({"split": split, "document": path.name, "error": str(exc)})
            continue
        total_pages += page_count
        output_dir = output_root / split / path.stem
        manifest_path = output_dir / "manifest.json"
        expected = _new_manifest(split, path, page_count, config, render_scale)
        existing = _load_manifest(manifest_path)
        manifest = existing if existing and _manifest_matches(existing, expected) else expected
        state = DocumentState(split, path, output_dir, manifest_path, manifest)
        states.append(state)
        for page in range(1, page_count + 1):
            page_key = str(page)
            record = manifest.get("pages", {}).get(page_key, {})
            page_file = output_dir / f"page-{page:04d}.txt"
            complete = (
                not force
                and record.get("status") == "completed"
                and record.get("file") == page_file.name
                and page_file.is_file()
                and page_file.stat().st_size > 0
            )
            if complete:
                skipped_pages += 1
                continue
            if max_pages is not None and len(tasks) >= max_pages:
                continue
            tasks.append(PageTask(state, page))
    return PrecomputePlan(states, tasks, errors, total_pages, skipped_pages)


async def _write_manifest(state: DocumentState) -> None:
    _refresh_counts(state.manifest)
    await asyncio.to_thread(
        atomic_write_text,
        state.manifest_path,
        json.dumps(state.manifest, ensure_ascii=False, indent=2) + "\n",
    )


async def _process_page(
    task: PageTask,
    client: OpenAICompatibleClient,
    document_service: DocumentService,
    semaphore: asyncio.Semaphore,
    *,
    render_scale: float,
    retries: int = 3,
) -> bool:
    state = task.document
    page_key = str(task.page)
    page_file = state.output_dir / f"page-{task.page:04d}.txt"
    async with semaphore:
        previous_attempts = int(state.manifest.get("pages", {}).get(page_key, {}).get("attempts") or 0)
        async with state.lock:
            state.manifest.setdefault("pages", {})[page_key] = {
                "status": "running",
                "file": page_file.name,
                "attempts": previous_attempts,
                "updated_at": utc_now(),
            }
            await _write_manifest(state)
        last_error: AppError | Exception | None = None
        attempts_used = 0
        for attempt in range(1, retries + 1):
            attempts_used = attempt
            started = time.perf_counter()
            try:
                image = await asyncio.to_thread(
                    document_service.render_page, state.path, task.page, render_scale
                )
                result = await client.chat(
                    f"请完整识别工业技术文档 {state.path.name} 的 PDF 第 {task.page} 页。",
                    OCR_SYSTEM_PROMPT,
                    image,
                    max_tokens=6400,
                )
                text = result["content"].strip()
                if not text:
                    raise AppError("MODEL_EMPTY_RESPONSE", "OCR 返回空文本", 502, True)
                await asyncio.to_thread(atomic_write_text, page_file, text + "\n")
                async with state.lock:
                    state.manifest["pages"][page_key] = {
                        "status": "completed",
                        "file": page_file.name,
                        "attempts": previous_attempts + attempts_used,
                        "characters": len(text),
                        "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                        "model": result.get("model") or client.config.model,
                        "updated_at": utc_now(),
                    }
                    await _write_manifest(state)
                print(f"OK {state.split}/{state.path.name} P.{task.page} chars={len(text)}", flush=True)
                await asyncio.sleep(INTER_PAGE_DELAY_SECONDS)
                return True
            except Exception as exc:
                last_error = exc
                retryable = not isinstance(exc, AppError) or exc.retryable
                if attempt < retries and retryable:
                    await asyncio.sleep((5, 15, 30)[min(attempt - 1, 2)])
                    continue
                break
        code = last_error.code if isinstance(last_error, AppError) else type(last_error).__name__
        message = last_error.message if isinstance(last_error, AppError) else str(last_error)
        async with state.lock:
            state.manifest["pages"][page_key] = {
                "status": "failed",
                "file": page_file.name,
                "attempts": previous_attempts + attempts_used,
                "error": {"code": code, "message": message[:500]},
                "updated_at": utc_now(),
            }
            await _write_manifest(state)
        print(f"FAIL {state.split}/{state.path.name} P.{task.page} code={code}", flush=True)
        await asyncio.sleep(INTER_PAGE_DELAY_SECONDS)
        return False


async def run(
    plan: PrecomputePlan,
    config: ModelEndpointConfig,
    *,
    concurrency: int,
    render_scale: float = 1.8,
    dry_run: bool = False,
) -> dict[str, Any]:
    summary = {
        "documents": len(plan.documents),
        "dataset_pages": plan.total_pages,
        "planned_pages": len(plan.tasks),
        "skipped_pages": plan.skipped_pages,
        "planning_errors": len(plan.planning_errors),
        "completed": 0,
        "failed": 0,
        "dry_run": dry_run,
    }
    if dry_run:
        return summary
    for state in plan.documents:
        state.output_dir.mkdir(parents=True, exist_ok=True)
        await _write_manifest(state)
    client = OpenAICompatibleClient(config, "ocr")
    service = DocumentService()
    semaphore = asyncio.Semaphore(concurrency)
    results = await asyncio.gather(
        *(
            _process_page(
                task,
                client,
                service,
                semaphore,
                render_scale=render_scale,
            )
            for task in plan.tasks
        )
    )
    summary["completed"] = sum(results)
    summary["failed"] = len(results) - summary["completed"]
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="预计算复赛 PDF 逐页 OCR 语料，不使用数据库。")
    parser.add_argument("--dataset-root", default=str(DATASET_ROOT), help="待识别数据集根目录，支持 preliminary_dataset 或 semi_final_dataset")
    parser.add_argument("--split", choices=("train", "test", "all"), default="all")
    parser.add_argument("--document", action="append", default=[], help="PDF 文件名或 stem；可重复或逗号分隔")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--force", action="store_true", help="忽略已完成页并重新 OCR")
    parser.add_argument("--dry-run", action="store_true", help="只输出计划，不调用模型、不写文件")
    parser.add_argument("--max-pages", type=int, default=None, help="本次最多处理的总页数")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not 1 <= args.concurrency <= 16:
        raise SystemExit("--concurrency 必须在 1 到 16 之间")
    if args.max_pages is not None and args.max_pages < 1:
        raise SystemExit("--max-pages 必须大于 0")
    config = load_ocr_config()
    splits = ("train", "test") if args.split == "all" else (args.split,)
    documents = discover_documents(
        dataset_root=Path(args.dataset_root).resolve(),
        splits=splits,
        selectors=args.document,
    )
    plan = build_plan(
        documents,
        config,
        force=args.force,
        max_pages=args.max_pages,
    )
    summary = asyncio.run(
        run(plan, config, concurrency=args.concurrency, dry_run=args.dry_run)
    )
    # Summary deliberately contains no endpoint or credential fields.
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if plan.planning_errors:
        print(json.dumps({"planning_errors": plan.planning_errors[:20]}, ensure_ascii=False, indent=2), flush=True)
    return 1 if summary["failed"] or summary["planning_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
