from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
WORKER = ROOT_DIR / "scripts" / "precompute_semifinal_ocr.py"
RUNTIME_DIR = ROOT_DIR / "runtime"
PID_FILE = RUNTIME_DIR / "ocr-precompute.pid"
LOG_FILE = RUNTIME_DIR / "ocr-precompute.log"
ERROR_LOG_FILE = RUNTIME_DIR / "ocr-precompute-error.log"


def process_running(pid: int) -> bool:
    if pid < 1:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
        if not handle:
            return False
        try:
            exit_code = wintypes.DWORD()
            return bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))) and exit_code.value == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def current_pid() -> int | None:
    try:
        pid = int(PID_FILE.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    return pid if process_running(pid) else None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在后台启动复赛 PDF MiniMax OCR 预计算任务。")
    parser.add_argument("--dataset-root", default=str(ROOT_DIR / "semi_final_dataset"), help="待识别数据集根目录")
    parser.add_argument("--status", action="store_true", help="只查看后台任务状态")
    parser.add_argument("--split", choices=("train", "test", "all"), default="all")
    parser.add_argument("--document", action="append", default=[], help="PDF 文件名或 stem，可重复传入")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def status_payload() -> dict[str, object]:
    pid = current_pid()
    return {
        "running": pid is not None,
        "pid": pid,
        "log": str(LOG_FILE.relative_to(ROOT_DIR)),
        "error_log": str(ERROR_LOG_FILE.relative_to(ROOT_DIR)),
        "output": "runtime/ocr-corpus/{train|test}/{pdf_stem}/page-0001.txt",
    }


def build_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(WORKER),
        "--dataset-root",
        args.dataset_root,
        "--split",
        args.split,
        "--concurrency",
        str(args.concurrency),
    ]
    for document in args.document:
        command.extend(("--document", document))
    if args.max_pages is not None:
        command.extend(("--max-pages", str(args.max_pages)))
    if args.force:
        command.append("--force")
    return command


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.status:
        print(json.dumps(status_payload(), ensure_ascii=False, indent=2))
        return 0
    if not 1 <= args.concurrency <= 8:
        raise SystemExit("--concurrency 必须在 1 到 8 之间；MiniMax OCR 建议使用 1。")
    existing = current_pid()
    if existing is not None:
        print(json.dumps({**status_payload(), "message": "OCR 后台任务已在运行"}, ensure_ascii=False, indent=2))
        return 0

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    creationflags = 0
    popen_kwargs: dict[str, object] = {"cwd": ROOT_DIR, "close_fds": True}
    if os.name == "nt":
        creationflags = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
            | getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
            | getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        )
        popen_kwargs["creationflags"] = creationflags
    else:
        popen_kwargs["start_new_session"] = True

    with LOG_FILE.open("ab") as output, ERROR_LOG_FILE.open("ab") as error:
        process = subprocess.Popen(build_command(args), stdout=output, stderr=error, **popen_kwargs)
    PID_FILE.write_text(f"{process.pid}\n", encoding="utf-8")
    print(
        json.dumps(
            {
                **status_payload(),
                "running": True,
                "pid": process.pid,
                "message": "MiniMax OCR 已在后台启动；已完成页面会自动跳过。",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
