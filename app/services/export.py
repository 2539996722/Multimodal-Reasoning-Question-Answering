from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Iterable

from app.config import ROOT_DIR


class ExportService:
    def __init__(self, export_dir: Path | str | None = None) -> None:
        self.export_dir = Path(export_dir or ROOT_DIR / "runtime" / "exports")
        self._lock = Lock()

    @staticmethod
    def render_jsonl(results: Iterable[dict[str, Any]]) -> bytes:
        lines = []
        for result in results:
            answer = result.get("answer")
            if answer is None and isinstance(result.get("result"), dict):
                answer = result["result"].get("answer")
            lines.append(json.dumps({"id": result.get("id"), "answer": answer or ""}, ensure_ascii=False))
        content = "\n".join(lines)
        return (content + ("\n" if content else "")).encode("utf-8")

    def write_jsonl(self, stage: str, results: Iterable[dict[str, Any]]) -> Path:
        payload = self.render_jsonl(results)
        with self._lock:
            self.export_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            target = self.export_dir / f"{stage}_{stamp}_{uuid.uuid4().hex[:8]}.jsonl"
            target.write_bytes(payload)
        return target
