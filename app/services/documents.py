from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from app.errors import AppError

try:
    import pymupdf
except ImportError:  # pragma: no cover - deployment capability branch
    pymupdf = None


class DocumentService:
    def metadata(self, path: Path) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "page_count": None,
            "render_available": pymupdf is not None,
        }
        if pymupdf is None:
            return metadata
        try:
            with pymupdf.open(path) as pdf:
                metadata["page_count"] = pdf.page_count
                metadata["pdf_metadata"] = {
                    key: value for key, value in (pdf.metadata or {}).items() if value
                }
        except Exception as exc:
            raise AppError("DOCUMENT_INVALID", "PDF 文件无法打开", 422, False) from exc
        return metadata

    def render_page(self, path: Path, page: int, scale: float = 1.6) -> bytes:
        if pymupdf is None:
            raise AppError("PDF_RENDER_UNAVAILABLE", "当前服务未安装 PDF 页面渲染能力", 503)
        if page < 1:
            raise AppError("PAGE_OUT_OF_RANGE", "页码必须从 1 开始", 400)
        return self._render_cached(str(path), path.stat().st_mtime_ns, page, round(scale, 2))

    @staticmethod
    @lru_cache(maxsize=48)
    def _render_cached(path_value: str, modified_ns: int, page: int, scale: float) -> bytes:
        del modified_ns
        try:
            with pymupdf.open(path_value) as pdf:
                if page > pdf.page_count:
                    raise AppError("PAGE_OUT_OF_RANGE", f"页码超出范围，文档共 {pdf.page_count} 页", 400)
                pixmap = pdf.load_page(page - 1).get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
                return pixmap.tobytes("png")
        except AppError:
            raise
        except Exception as exc:
            raise AppError("PAGE_RENDER_FAILED", "PDF 页面渲染失败", 422, True) from exc
