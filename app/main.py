from __future__ import annotations

import mimetypes
import uuid
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import __version__
from app.api.routes import router
from app.config import ROOT_DIR
from app.errors import AppError, app_error_handler, error_payload, validation_error_handler


# 自托管字体需要正确的 MIME 类型，否则部分浏览器/代理会拒绝加载。
mimetypes.add_type("font/woff2", ".woff2")
mimetypes.add_type("font/woff", ".woff")

app = FastAPI(
    title="InduDoc AI API",
    description="工业文档智能解析与评测平台",
    version=__version__,
    docs_url="/api/docs",
    redoc_url=None,
    openapi_url="/api/openapi.json",
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    request.state.request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    started_at = perf_counter()
    response = await call_next(request)
    duration_ms = (perf_counter() - started_at) * 1000
    response.headers["X-Request-ID"] = request.state.request_id
    response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"
    response.headers["Server-Timing"] = f'app;dur={duration_ms:.2f}'
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["Referrer-Policy"] = "same-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=(), payment=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; base-uri 'self'; form-action 'self'; frame-ancestors 'self'; "
        "object-src 'self' blob:; img-src 'self' data: blob:; font-src 'self' data:; "
        "style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'"
    )
    if request.url.scheme == "https":
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
    return response


app.add_exception_handler(AppError, app_error_handler)
app.add_exception_handler(RequestValidationError, validation_error_handler)


@app.exception_handler(StarletteHTTPException)
async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    code = "NOT_FOUND" if exc.status_code == 404 else "HTTP_ERROR"
    message = "未找到请求的资源" if exc.status_code == 404 else str(exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content=error_payload(code, message, request.state.request_id, False),
    )


@app.exception_handler(Exception)
async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    del exc
    return JSONResponse(
        status_code=500,
        content=error_payload("INTERNAL_ERROR", "服务处理请求时发生内部错误", request.state.request_id, True),
    )


app.include_router(router)

static_dir = ROOT_DIR / "app" / "static"
if static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", include_in_schema=False)
async def index():
    index_path = static_dir / "index.html"
    if index_path.is_file():
        return FileResponse(index_path, media_type="text/html")
    return {"service": "InduDoc AI", "api_docs": "/api/docs"}
