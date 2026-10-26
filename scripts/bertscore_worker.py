#!/usr/bin/env python
"""BERTScore 计算子进程。

为什么必须独立进程：bert_score 会加载 torch。本机上 torch 与 faiss 同进程共存时，
检索链路会发生原生崩溃（进程直接退出、exit=3、无任何 traceback），因此
主进程（scripts/evaluate_regression.py 与 pytest）一律不得导入 bert_score / torch，
BERTScore 只能在这个子进程里计算。

协议（stdin -> stdout）：
  stdin  : {"pairs": [[prediction, reference], ...]}
  stdout : {"f1": [float, ...], "model_type": "bert-base-chinese", "lang": "zh"}
  stderr : 诊断信息（模型下载进度、失败原因）
失败时以非 0 退出码结束，并在 stderr 写明原因。

只有本文件允许导入 bert_score / torch。
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from contextlib import redirect_stdout
from typing import Any


MODEL_TYPE = "bert-base-chinese"
LANG = "zh"


def force_utf8_streams() -> None:
    """强制子进程用 UTF-8 读写。

    Windows 控制台默认编码是 cp936：父进程写出的 UTF-8 载荷会被子进程按 cp936 解码，
    非法字节还会被 surrogateescape 变成孤立代理字符，PyO3 随即拒绝该字符串
    （tokenizers 报 "TextEncodeInput must be Union[...]"）。这里统一按 UTF-8 处理。
    """

    for stream in (sys.stdin, sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover - 少数不可重配的流
            pass


def apply_offline_mode(offline: bool) -> None:
    """离线模式：只用本地 HuggingFace 缓存。

    hub 不可达时，transformers 会为一次普通的缓存命中反复重试连接（每次 120s 超时），
    评测环境下应直接走缓存。必须在导入 bert_score / transformers 之前设置环境变量。
    """

    if not offline:
        return
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def read_pairs(payload: Any) -> tuple[list[str], list[str]] | None:
    """校验 stdin 载荷；不合法返回 None。"""

    if not isinstance(payload, dict):
        return None
    pairs = payload.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        return None
    predictions: list[str] = []
    references: list[str] = []
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            return None
        predictions.append(str(pair[0]))
        references.append(str(pair[1]))
    return predictions, references


def main() -> int:
    force_utf8_streams()
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"BERTScore worker: stdin 不是合法 JSON：{exc}", file=sys.stderr)
        return 2

    pairs = read_pairs(payload)
    if pairs is None:
        print("BERTScore worker: 缺少合法的非空 pairs 列表", file=sys.stderr)
        return 2
    predictions, references = pairs
    offline = bool(payload.get("offline")) if isinstance(payload, dict) else False
    apply_offline_mode(offline)

    try:
        import bert_score
    except ImportError as exc:
        print(f"BERTScore worker: 无法导入 bert_score：{exc}", file=sys.stderr)
        return 3

    try:
        # 依赖库可能往 stdout 打印下载进度，评测脚本要求 stdout 只有 JSON
        with redirect_stdout(sys.stderr):
            _, _, f1 = bert_score.score(
                predictions,
                references,
                model_type=MODEL_TYPE,
                lang=LANG,
            )
    except Exception as exc:  # noqa: BLE001 - 失败原因必须显式回传主进程
        traceback.print_exc(file=sys.stderr)
        print(f"BERTScore worker: 计算失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 4

    result = {
        "f1": [float(value) for value in f1],
        "model_type": MODEL_TYPE,
        "lang": LANG,
        "count": len(predictions),
        "offline": offline,
    }
    json.dump(result, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())