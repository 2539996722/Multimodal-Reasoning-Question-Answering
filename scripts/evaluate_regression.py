#!/usr/bin/env python
"""复赛回归评测与 A/B 对照工具（InduDoc AI）。

四层结构，彼此独立、可单测：
  1. 选集层  select_regression_set：按 PDF 分组选文档 -> 文档内类别均衡贪心选题 -> 输出按 PDF 聚拢。
  2. 执行层  run_arm / call_with_retry：HTTP 调用 POST /api/analyze，仅对 retryable=true 退避重试。
  3. 指标层  answer_metrics / retrieval_metrics：复用 app.services.evaluation 的规范化 EM 与 jieba ROUGE-1 F1。
  4. 报告层  build_report / write_report：report.json（机器可读）+ report.md（人读）。

BERTScore 是可选指标，且只在独立子进程 scripts/bertscore_worker.py 里计算：
bert_score 会加载 torch，torch 与 faiss 同进程共存会导致检索链路原生崩溃
（进程直接退出、exit=3、无 traceback），所以主进程绝不导入 bert_score / torch。

题目目录来自后端同一份 DatasetRepository 索引（app.services.dataset），字段与 GET /api/questions
完全一致（id/question/document/answer/page_hint/has_reference），但完全离线且可复现：
--dry-run 不发起任何 HTTP 请求；HTTP 只发生在执行层的 /api/analyze 调用上。

用法：
    python scripts/evaluate_regression.py --help
    python scripts/evaluate_regression.py --dry-run --size 8 --seed 20261025
    python scripts/evaluate_regression.py --size 8 --seed 20261025 --arms baseline,rag
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import re
import statistics
import subprocess
import sys
import time
import unicodedata
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import httpx

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.errors import AppError  # noqa: E402
from app.services.dataset import DatasetRepository  # noqa: E402
from app.services.evaluation import evaluate_answer, final_score  # noqa: E402


DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_OUTPUT_ROOT = ROOT_DIR / "runtime" / "regression"
DEFAULT_TIMEOUT_SECONDS = 180.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_BACKOFF_SECONDS = 1.5
DEFAULT_SIZE = 8
DEFAULT_SEED = 20261025
MAX_CONCURRENCY = 4
BERTSCORE_WORKER = Path(__file__).resolve().parent / "bertscore_worker.py"
DEFAULT_BERTSCORE_TIMEOUT = 900.0
RECALL_KS = (1, 3, 6)
RETRYABLE_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})

ARM_LABELS = {
    "baseline": "baseline（目标页 OCR + VLM，use_rag=false）",
    "rag": "rag（LangChain 分块 -> FAISS 召回 -> Reranker 重排 -> VLM，use_rag=true）",
}
ARM_USE_RAG = {"baseline": False, "rag": True}

# 四类人工抽检题型。优先级自上而下，用于给每个题目一个唯一的主分类。
CATEGORY_PATTERNS: dict[str, re.Pattern[str]] = {
    "spatial_relation": re.compile(
        r"位置关系|相对位置|位于|左侧|右侧|上方|下方|之间|相邻|对应|图中|示意图|标号|编号"
    ),
    "connection_state": re.compile(
        r"连接|相连|配合|固定|安装|啮合|连通|状态|打开|关闭|运动|驱动|装配|接合"
    ),
    "function_problem": re.compile(
        r"作用|功能|用途|目的|效果|优点|用于|实现|原理|解决|具备|特征"
    ),
    "process_causal": re.compile(
        r"步骤|过程|顺序|首先|然后|接着|之后|如何|为什么|原因|导致|流程|方法"
    ),
}
CATEGORY_ORDER = tuple(CATEGORY_PATTERNS)
CATEGORY_LABELS = {
    "spatial_relation": "空间关系",
    "connection_state": "部件连接/状态",
    "function_problem": "功能/技术问题",
    "process_causal": "工艺步骤/因果",
    "other": "未归类",
}
UNCATEGORIZED = "other"


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def force_utf8_output() -> None:
    """中文报告在 GBK 控制台下也要可读，统一把标准输出切到 UTF-8。"""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover - 少数不可重配的流
            pass


def normalize_document_key(value: str) -> str:
    """文档名归一：忽略大小写与 .pdf 后缀，便于 --exclude-documents 精确互斥。"""

    text = (value or "").strip().casefold()
    if text.endswith(".pdf"):
        text = text[: -len(".pdf")]
    return text


def parse_target_page(question: str) -> int | None:
    """从题干解析「第 N 页」提示；解析不出返回 None（该题不计入检索指标）。

    直接复用后端 DatasetRepository.extract_page_hint，避免同一套「第 N 页」正则出现两份。
    """

    return DatasetRepository.extract_page_hint(question or "")


def match_categories(question: str) -> list[str]:
    """返回题干命中的全部题型（可多命中）。"""

    text = question or ""
    return [name for name in CATEGORY_ORDER if CATEGORY_PATTERNS[name].search(text)]


def classify_question(question: str) -> str:
    """给题干一个唯一主分类（按固定优先级取第一个命中）。"""

    hits = match_categories(question)
    return hits[0] if hits else UNCATEGORIZED


def percentile(values: Sequence[float], ratio: float) -> float | None:
    """线性插值分位数（ratio 取 0-1）。"""

    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * ratio
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 2)


def mean_or_none(values: Iterable[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    if not valid:
        return None
    return round(statistics.fmean(valid), 6)


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# 1. 选集层
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class QuestionItem:
    """一道题的全部选集与评测所需字段。"""

    id: str
    question: str
    document: str
    answer: str | None
    page_hint: int | None
    has_reference: bool
    category: str
    target_page: int | None

    @property
    def has_target_page(self) -> bool:
        return self.target_page is not None


@dataclass
class Selection:
    """固定回归集：题目、按 PDF 聚拢的分组信息与题型分布。"""

    items: list[QuestionItem]
    groups: list[dict[str, Any]]
    excluded_documents: list[str]
    available_total: int
    available_documents: int
    primary_category_counts: dict[str, int]
    pattern_match_counts: dict[str, int]
    seed: int
    requested_size: int

    @property
    def documents(self) -> list[str]:
        return [group["document"] for group in self.groups]


def load_catalog(stage: str, split: str) -> list[QuestionItem]:
    """加载分区题目目录（复用后端 DatasetRepository，与 /api/questions 字段一致）。"""

    repository = DatasetRepository(ROOT_DIR)
    partition = repository.partition(stage, split)
    return [
        QuestionItem(
            id=str(record.id),
            question=record.question,
            document=record.document,
            answer=record.answer,
            page_hint=record.page_hint,
            has_reference=record.has_reference,
            category=classify_question(record.question),
            target_page=parse_target_page(record.question),
        )
        for record in partition.records
    ]


def _sort_key(item: QuestionItem) -> tuple[int, Any]:
    text = item.id.strip()
    return (0, int(text)) if text.isdigit() else (1, text)


def group_by_document(items: Iterable[QuestionItem]) -> dict[str, list[QuestionItem]]:
    groups: dict[str, list[QuestionItem]] = {}
    for item in items:
        groups.setdefault(item.document, []).append(item)
    for value in groups.values():
        value.sort(key=_sort_key)
    return groups


def _pick_documents(
    candidates: Sequence[str],
    groups: dict[str, list[QuestionItem]],
    size: int,
) -> list[str]:
    """先选 PDF：先让四类题型各有落点，再按可用题量补足。"""

    categories = {document: {item.category for item in items} for document, items in groups.items()}
    picked: list[str] = []
    for category in CATEGORY_ORDER:
        for document in candidates:
            if document in picked:
                continue
            if category in categories[document]:
                picked.append(document)
                break
    available = sum(len(groups[document]) for document in picked)
    for document in candidates:
        if available >= size:
            break
        if document in picked:
            continue
        picked.append(document)
        available += len(groups[document])
    return picked


def _pick_next_question(
    queues: dict[str, list[QuestionItem]],
    order: Sequence[str],
    counts: Counter[str],
) -> tuple[str, QuestionItem] | None:
    """全局贪心：优先取「当前已选题量最少」的类别，其余按文档顺序、队列顺序决胜。"""

    best_key: tuple[int, int, int] | None = None
    best: tuple[str, QuestionItem] | None = None
    for document_index, document in enumerate(order):
        for queue_index, item in enumerate(queues[document]):
            key = (counts.get(item.category, 0), document_index, queue_index)
            if best_key is None or key < best_key:
                best_key = key
                best = (document, item)
    return best


def select_regression_set(
    items: Sequence[QuestionItem],
    size: int,
    seed: int,
    exclude_documents: Sequence[str] = (),
) -> Selection:
    """按 PDF 分组选固定回归集。

    1) 先选 PDF：排除指定文档后按 seed 打乱文档顺序；先让空间关系 / 连接状态 /
       功能问题 / 工艺步骤四类题型各有落点，再按可用题量补足到 size。
    2) 再选题：在选中的 PDF 内全局贪心取题，每次优先取「当前已选题量最少」的类别，
       保证四类题型尽量均衡覆盖。
    3) 输出顺序按 PDF 聚拢：同一 PDF 的题连续出现，任何按位置切分的分组都不会把
       同一 PDF 拆到两组（调参集与验证集互斥的前提）。
    """

    if size <= 0:
        raise ValueError("--size 必须为正整数")

    excluded_keys = {normalize_document_key(value) for value in exclude_documents if value and value.strip()}
    groups = group_by_document(items)
    candidates = [
        document
        for document in sorted(groups)
        if normalize_document_key(document) not in excluded_keys
    ]

    rng = random.Random(seed)
    rng.shuffle(candidates)

    picked_documents = _pick_documents(candidates, groups, size)

    queues = {document: list(groups[document]) for document in picked_documents}
    counts: Counter[str] = Counter()
    chosen: dict[str, list[QuestionItem]] = {document: [] for document in picked_documents}
    total = 0
    while total < size:
        picked = _pick_next_question(queues, picked_documents, counts)
        if picked is None:
            break
        document, item = picked
        queues[document].remove(item)
        counts[item.category] += 1
        chosen[document].append(item)
        total += 1

    ordered: list[QuestionItem] = []
    group_report: list[dict[str, Any]] = []
    cursor = 0
    for document in picked_documents:
        block = chosen[document]
        if not block:
            continue
        ordered.extend(block)
        group_report.append(
            {
                "document": document,
                "count": len(block),
                "start_index": cursor,
                "end_index": cursor + len(block) - 1,
            }
        )
        cursor += len(block)

    primary_counts = Counter(item.category for item in ordered)
    pattern_counts: Counter[str] = Counter()
    for item in ordered:
        for name in match_categories(item.question):
            pattern_counts[name] += 1

    return Selection(
        items=ordered,
        groups=group_report,
        excluded_documents=sorted(exclude_documents),
        available_total=len(items),
        available_documents=len(groups),
        primary_category_counts={
            **{name: primary_counts.get(name, 0) for name in CATEGORY_ORDER},
            UNCATEGORIZED: primary_counts.get(UNCATEGORIZED, 0),
        },
        pattern_match_counts={name: pattern_counts.get(name, 0) for name in CATEGORY_ORDER},
        seed=seed,
        requested_size=size,
    )


def selection_distribution(selection: Selection, key: str) -> dict[str, dict[str, Any]]:
    total = len(selection.items) or 1
    counts = selection.primary_category_counts if key == "primary" else selection.pattern_match_counts
    return {
        name: {"count": count, "ratio": round(count / total, 4)}
        for name, count in counts.items()
    }


# --------------------------------------------------------------------------- #
# 2. HTTP 客户端与执行层
# --------------------------------------------------------------------------- #
class ApiError(Exception):
    """后端统一错误响应或传输层失败。"""

    def __init__(
        self,
        code: str,
        message: str,
        retryable: bool,
        status_code: int | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.retryable = bool(retryable)
        self.status_code = status_code
        self.request_id = request_id
        self.attempts = 1


def parse_error_response(status_code: int, payload: Any) -> ApiError:
    """把后端 {"error": {...}} 响应映射为 ApiError；缺失 retryable 时按状态码推断。"""

    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict):
        code = str(error.get("code") or f"HTTP_{status_code}")
        message = str(error.get("message") or f"HTTP {status_code}")
        retryable = bool(error.get("retryable", status_code in RETRYABLE_STATUS_CODES))
        request_id = error.get("request_id")
    else:
        code = f"HTTP_{status_code}"
        message = f"HTTP {status_code}"
        retryable = status_code in RETRYABLE_STATUS_CODES
        request_id = None
    return ApiError(code, message, retryable, status_code, request_id)


class HttpApiClient:
    """InduDoc AI 后端的最小 HTTP 客户端（仅调用 POST /api/analyze）。"""

    def __init__(
        self,
        base_url: str,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout, transport=transport)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "HttpApiClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def analyze(
        self,
        *,
        stage: str,
        split: str,
        question_id: str,
        use_rag: bool,
        force_demo: bool,
        use_ocr: bool = True,
    ) -> dict[str, Any]:
        payload = {
            "stage": stage,
            "split": split,
            "question_id": question_id,
            "force_demo": force_demo,
            "use_ocr": use_ocr,
            "use_rag": use_rag,
        }
        try:
            response = self._client.post("/api/analyze", json=payload)
        except httpx.HTTPError as exc:
            raise ApiError("TRANSPORT_ERROR", f"请求传输失败：{exc}", True) from exc
        if response.status_code >= 400:
            try:
                body = response.json()
            except ValueError:
                body = None
            raise parse_error_response(response.status_code, body)
        try:
            return response.json()
        except ValueError as exc:
            raise ApiError("INVALID_RESPONSE", f"服务返回了非 JSON 响应：{exc}", False) from exc


@dataclass
class ArmOutcome:
    """单题单臂的执行结果（成功或失败都保留显式信息）。"""

    question_id: str
    document: str
    ok: bool
    attempts: int
    answer: str | None = None
    duration_ms: float | None = None
    trace_duration_ms: float | None = None
    mode: str | None = None
    error: dict[str, Any] | None = None
    rag: dict[str, Any] | None = None
    evidence_pages: list[int] = field(default_factory=list)
    backend_metrics: dict[str, Any] | None = None


def call_with_retry(
    operation: Callable[[], dict[str, Any]],
    *,
    max_retries: int,
    backoff_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
    on_retry: Callable[[int, ApiError], None] | None = None,
) -> tuple[dict[str, Any], int]:
    """仅对 retryable=true 的错误做退避重试，返回 (结果, 实际尝试次数)。"""

    total_attempts = max(0, max_retries) + 1
    for attempt in range(1, total_attempts + 1):
        try:
            return operation(), attempt
        except ApiError as exc:
            exc.attempts = attempt
            if not exc.retryable or attempt >= total_attempts:
                raise
            if on_retry is not None:
                on_retry(attempt, exc)
            sleep(backoff_seconds * attempt)
    raise RuntimeError("重试流程异常结束")  # pragma: no cover - 逻辑上不可达


def unwrap_result(payload: dict[str, Any]) -> dict[str, Any]:
    """兼容两种响应外壳：{"result": {...}} 与扁平结构。"""

    result = payload.get("result")
    return result if isinstance(result, dict) else payload


def extract_trace(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """从 trace / technical_trace / pipeline 中取出阶段数组。"""

    result = unwrap_result(payload)
    for key in ("trace", "technical_trace", "pipeline"):
        value = payload.get(key)
        if not isinstance(value, list):
            value = result.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def trace_duration_ms(trace: Sequence[dict[str, Any]]) -> float | None:
    """各阶段耗时求和（duration_ms 缺失时作为兜底口径）。"""

    total = 0.0
    found = False
    for item in trace:
        value = item.get("duration_ms")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            total += float(value)
            found = True
    return round(total, 1) if found else None


def build_outcome(item: QuestionItem, payload: dict[str, Any], attempts: int) -> ArmOutcome:
    result = unwrap_result(payload)
    trace = extract_trace(payload)
    duration = result.get("duration_ms")
    if not isinstance(duration, (int, float)) or isinstance(duration, bool):
        duration = trace_duration_ms(trace)
    rag = result.get("rag")
    evidence_pages = [page for page in (result.get("evidence_pages") or []) if isinstance(page, int)]
    backend_metrics = result.get("metrics")
    return ArmOutcome(
        question_id=item.id,
        document=item.document,
        ok=True,
        attempts=attempts,
        answer=str(result.get("answer") or "").strip() or None,
        duration_ms=float(duration) if isinstance(duration, (int, float)) else None,
        trace_duration_ms=trace_duration_ms(trace),
        mode=str(result.get("mode")) if result.get("mode") else None,
        rag=rag if isinstance(rag, dict) else None,
        evidence_pages=evidence_pages,
        backend_metrics=backend_metrics if isinstance(backend_metrics, dict) else None,
    )


def run_arm(
    client: Any,
    arm: str,
    items: Sequence[QuestionItem],
    *,
    stage: str,
    split: str,
    force_demo: bool,
    concurrency: int = 1,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    use_ocr: bool = True,
    sleep: Callable[[float], None] = time.sleep,
    on_progress: Callable[[str, int, int, ArmOutcome], None] | None = None,
) -> list[ArmOutcome]:
    """在一个臂上顺序执行固定题集；单题失败不终止整批。"""

    if arm not in ARM_USE_RAG:
        raise ValueError(f"未知评测臂：{arm}")
    use_rag = ARM_USE_RAG[arm]
    workers = max(1, min(int(concurrency), MAX_CONCURRENCY))

    def execute(index: int, item: QuestionItem) -> ArmOutcome:
        def operation() -> dict[str, Any]:
            return client.analyze(
                stage=stage,
                split=split,
                question_id=item.id,
                use_rag=use_rag,
                force_demo=force_demo,
                use_ocr=use_ocr,
            )

        def on_retry(attempt: int, exc: ApiError) -> None:
            print(
                f"[{arm}] 第 {attempt} 次调用失败（{exc.code}，可重试），准备重试 id={item.id}",
                flush=True,
            )

        try:
            payload, attempts = call_with_retry(
                operation,
                max_retries=max_retries,
                backoff_seconds=backoff_seconds,
                sleep=sleep,
                on_retry=on_retry,
            )
        except ApiError as exc:
            outcome = ArmOutcome(
                question_id=item.id,
                document=item.document,
                ok=False,
                attempts=exc.attempts,
                error={
                    "code": exc.code,
                    "message": exc.message,
                    "retryable": exc.retryable,
                    "status_code": exc.status_code,
                    "request_id": exc.request_id,
                },
            )
        else:
            outcome = build_outcome(item, payload, attempts)
        if on_progress is not None:
            on_progress(arm, index, len(items), outcome)
        return outcome

    if workers == 1:
        return [execute(index, item) for index, item in enumerate(items)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda pair: execute(pair[0], pair[1]), list(enumerate(items))))

# --------------------------------------------------------------------------- #
# 3. 指标层
# --------------------------------------------------------------------------- #
def answer_metrics(
    stage: str,
    prediction: str | None,
    reference: str | None,
    bert_f1: float | None = None,
) -> dict[str, Any]:
    """复用 app.services.evaluation 的规范化口径，不在脚本里另起一套规范化。

    - exact_match：NFKC + casefold + 去首尾空白后的严格相等（与后端一致）。
    - rouge1_f1：rouge-score + jieba 分词（复赛官方口径）。
    - final_score：官方权重 0.5*BERTScore + 0.25*ROUGE-1 + 0.25*EM；缺任一项则为空。
    """

    evaluated = evaluate_answer(stage, prediction, reference)
    exact = evaluated.get("exact_match")
    rouge = evaluated.get("rouge1_f1")
    return {
        "accuracy": evaluated.get("accuracy"),
        "exact_match": exact,
        "rouge1_f1": rouge,
        "bert_f1": bert_f1,
        "final_score": final_score(bert_f1, rouge, exact),
    }


def retrieval_metrics(
    rag_items: Sequence[dict[str, Any]],
    target_page: int | None,
    *,
    enabled: bool = True,
) -> dict[str, Any]:
    """Recall@1/3/6 与 MRR：按 rag.items 的召回顺序判定目标页命中。

    未启用 RAG 的评测臂（例如 baseline）没有召回序列，检索指标记为「不适用」，
    不能记成 0（0 表示「召回了但没命中」，语义完全不同）。
    """

    pages = [item.get("page") for item in rag_items]
    if not enabled:
        return {
            "applicable": False,
            "skip_reason": "该评测臂未启用 RAG（use_rag=false），无召回序列",
            "target_page": target_page,
            "item_count": len(rag_items),
            "first_hit_rank": None,
            "mrr": None,
            "recall_at_1": None,
            "recall_at_3": None,
            "recall_at_6": None,
        }
    if target_page is None:
        return {
            "applicable": False,
            "skip_reason": "题干未包含「第 N 页」提示，无法确定目标页",
            "target_page": None,
            "item_count": len(rag_items),
            "first_hit_rank": None,
            "mrr": None,
            "recall_at_1": None,
            "recall_at_3": None,
            "recall_at_6": None,
        }
    ranks = [index + 1 for index, page in enumerate(pages) if page == target_page]
    first_hit = ranks[0] if ranks else None

    def hit(limit: int) -> float:
        return 1.0 if first_hit is not None and first_hit <= limit else 0.0

    return {
        "applicable": True,
        "skip_reason": None,
        "target_page": target_page,
        "item_count": len(rag_items),
        "first_hit_rank": first_hit,
        "mrr": round(1.0 / first_hit, 6) if first_hit else 0.0,
        **{f"recall_at_{limit}": hit(limit) for limit in RECALL_KS},
    }


def rag_degradation(rag: dict[str, Any] | None) -> tuple[bool, str | None]:
    """判定 RAG 是否降级；后端未直接给出 degraded 字段时按协议与索引状态推导。"""

    if not rag:
        return False, None
    if "degraded" in rag:
        reason = rag.get("reason")
        return bool(rag.get("degraded")), (str(reason) if reason else None)
    reasons: list[str] = []
    retrieval_protocol = str(rag.get("retrieval_protocol") or "")
    rerank_protocol = str(rag.get("rerank_protocol") or "")
    if retrieval_protocol.endswith("_degraded"):
        reasons.append(f"召回协议降级：{retrieval_protocol}")
    if rag.get("vector_index_status") == "degraded":
        reasons.append("向量索引降级")
    if rerank_protocol == "retrieval_score_fallback":
        reasons.append("重排降级：回退召回分排序")
    for key in ("degradation_reason", "rerank_degradation_reason"):
        value = rag.get(key)
        if not value:
            continue
        reasons.append(str(value.get("message") if isinstance(value, dict) else value))
    if not reasons:
        return False, None
    return True, "；".join(reasons)


class BertScoreRunner:
    """可选的 BERTScore F1 计算器（跑在独立子进程里）。

    为什么不直接在主进程 import bert_score：bert_score 会加载 torch，本机上
    torch 与 faiss 同进程共存会导致检索链路原生崩溃（进程直接退出、exit=3、无 traceback）。
    因此主进程只负责「探测 + 拉起子进程 + 解析结果」，永不导入 bert_score / torch。

    子进程不可用 / 超时 / 返回非法 JSON 时，明确报告「未计算」并写明原因，
    绝不用 0 或任何伪造值填补官方综合分。
    """

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.status = "未计算（未开启 --bertscore）"
        self.error: str | None = None
        self.worker = BERTSCORE_WORKER
        self.timeout = DEFAULT_BERTSCORE_TIMEOUT
        self.offline = False
        self.mode = ""
        self.model_type = ""
        self.lang = ""
        self._ready = False
        if not enabled:
            return
        if importlib.util.find_spec("bert_score") is None:
            self.status = "未计算（未安装 bert_score）"
            return
        if not self.worker.is_file():
            self.status = f"未计算（找不到 BERTScore 子进程：{self.worker.name}）"
            return
        self._ready = True
        self.status = "待计算"

    @property
    def available(self) -> bool:
        """只做探测（find_spec + 子进程文件存在），绝不真正 import bert_score。"""

        return self._ready

    def _fail(self, reason: str) -> dict[str, float]:
        self.error = reason
        self.status = f"未计算（{reason}）"
        print(f"[bertscore] {self.status}", flush=True)
        return {}

    def _run_worker(
        self,
        pairs: Sequence[tuple[str, str, str]],
        offline: bool,
    ) -> tuple[dict[str, float], str]:
        """跑一次子进程，返回 (分数, 失败原因)。"""

        payload = {
            "pairs": [[prediction, reference] for _, prediction, reference in pairs],
            "offline": bool(offline),
        }
        try:
            completed = subprocess.run(
                [sys.executable, str(self.worker)],
                input=json.dumps(payload, ensure_ascii=False),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {}, f"BERTScore 子进程超时（>{self.timeout:.0f}s）"
        except OSError as exc:
            return {}, f"无法启动 BERTScore 子进程：{exc}"

        if completed.returncode != 0:
            detail = (completed.stderr or "").strip().splitlines()
            reason = detail[-1] if detail else f"子进程退出码 {completed.returncode}"
            return {}, f"BERTScore 子进程失败：{reason}"

        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            return {}, f"BERTScore 子进程输出不是合法 JSON：{exc}"

        scores = result.get("f1") if isinstance(result, dict) else None
        if not isinstance(scores, list) or len(scores) != len(pairs):
            return {}, f"BERTScore 子进程返回的分数数量不匹配（期望 {len(pairs)}）"

        self.mode = "本地缓存模式" if offline else "在线模式"
        self.model_type = str(result.get("model_type") or "")
        self.lang = str(result.get("lang") or "")
        return {
            question_id: float(score)
            for (question_id, _, _), score in zip(pairs, scores)
        }, ""

    def score(self, pairs: Sequence[tuple[str, str, str]]) -> dict[str, float]:
        """pairs: (question_id, prediction, reference) -> {question_id: bert_f1}。"""

        if not self.enabled:
            return {}
        if not self.available:
            return {}
        if not pairs:
            return self._fail("没有可评分的答案对")

        attempts = [self.offline]
        if not self.offline:
            # hub 不可达但模型已缓存是评测环境最常见的失败原因，失败后再用纯缓存模式试一次
            attempts.append(True)
        reason = ""
        for offline in attempts:
            scores, reason = self._run_worker(pairs, offline)
            if scores:
                self.error = None
                self.status = (
                    f"已计算（{self.model_type} / {self.lang} · 独立子进程 · {self.mode}）"
                )
                return scores
        return self._fail(reason)


def build_question_records(
    stage: str,
    outcomes: Sequence[ArmOutcome],
    items_by_id: dict[str, QuestionItem],
    bert_scores: dict[str, float],
) -> list[dict[str, Any]]:
    """把执行结果与指标合并成逐题明细。"""

    records: list[dict[str, Any]] = []
    for outcome in outcomes:
        item = items_by_id[outcome.question_id]
        record: dict[str, Any] = {
            "id": outcome.question_id,
            "document": outcome.document,
            "category": item.category,
            "category_label": CATEGORY_LABELS.get(item.category, item.category),
            "target_page": item.target_page,
            "reference": item.answer,
            "prediction": outcome.answer,
            "ok": outcome.ok,
            "attempts": outcome.attempts,
            "duration_ms": outcome.duration_ms,
            "trace_duration_ms": outcome.trace_duration_ms,
            "mode": outcome.mode,
            "error": outcome.error,
            "evidence_pages": outcome.evidence_pages,
            "backend_metrics": outcome.backend_metrics,
        }
        if not outcome.ok:
            record.update(
                {
                    "accuracy": None,
                    "exact_match": None,
                    "rouge1_f1": None,
                    "bert_f1": None,
                    "final_score": None,
                    "retrieval": {
                        "applicable": False,
                        "skip_reason": "题目执行失败，无检索结果",
                        "target_page": item.target_page,
                        "item_count": 0,
                        "first_hit_rank": None,
                        "mrr": None,
                        "recall_at_1": None,
                        "recall_at_3": None,
                        "recall_at_6": None,
                    },
                    "rag": None,
                }
            )
            records.append(record)
            continue

        rag = outcome.rag or {}
        rag_items = [item for item in (rag.get("items") or []) if isinstance(item, dict)]
        degraded, degradation_reason = rag_degradation(rag)
        record.update(answer_metrics(stage, outcome.answer, item.answer, bert_scores.get(outcome.question_id)))
        record["retrieval"] = retrieval_metrics(
            rag_items, item.target_page, enabled=bool(rag.get("enabled"))
        )
        record["rag"] = {
            "enabled": rag.get("enabled"),
            "requested": rag.get("requested"),
            "chunk_count": rag.get("chunk_count"),
            "item_count": len(rag_items),
            "retrieval_protocol": rag.get("retrieval_protocol"),
            "rerank_protocol": rag.get("rerank_protocol"),
            "index_version": rag.get("index_version") or rag.get("vector_index_key"),
            "degraded": degraded,
            "degradation_reason": degradation_reason,
        }
        records.append(record)
    return records


def aggregate_arm(records: Sequence[dict[str, Any]], bert_status: str) -> dict[str, Any]:
    """把逐题明细汇总成臂级指标。"""

    succeeded = [record for record in records if record["ok"]]
    failed = [record for record in records if not record["ok"]]
    scored = [record for record in succeeded if record["reference"] is not None]
    retrieval_scored = [record for record in succeeded if record["retrieval"]["applicable"]]

    durations = [record["duration_ms"] for record in succeeded if record["duration_ms"] is not None]
    chunks = [
        record["rag"]["chunk_count"]
        for record in succeeded
        if record["rag"] and isinstance(record["rag"].get("chunk_count"), (int, float))
    ]
    fragments = [
        record["rag"]["item_count"]
        for record in succeeded
        if record["rag"] and isinstance(record["rag"].get("item_count"), int)
    ]
    degraded = [record for record in succeeded if record["rag"] and record["rag"]["degraded"]]

    retrieval_protocols = Counter(
        record["rag"]["retrieval_protocol"] for record in succeeded if record["rag"]
    )
    rerank_protocols = Counter(
        record["rag"]["rerank_protocol"] for record in succeeded if record["rag"]
    )
    modes = Counter(record["mode"] or "unknown" for record in succeeded)

    agreements: list[float] = []
    for record in succeeded:
        backend = record.get("backend_metrics") or {}
        backend_em = backend.get("exact_match")
        if backend_em is None or record["exact_match"] is None:
            continue
        agreements.append(1.0 if float(backend_em) == float(record["exact_match"]) else 0.0)

    final_score_value = mean_or_none(record["final_score"] for record in scored)
    bert_value = mean_or_none(record["bert_f1"] for record in scored)
    final_status = (
        "已计算"
        if final_score_value is not None
        else f"不完整（缺 BERTScore：{bert_status}）"
    )
    return {
        "totals": {
            "questions": len(records),
            "succeeded": len(succeeded),
            "failed": len(failed),
            "failure_rate": round(len(failed) / len(records), 6) if records else None,
            "with_reference": len(scored),
            "retrieval_applicable": len(retrieval_scored),
            "retrieval_skipped": len(succeeded) - len(retrieval_scored),
        },
        "answer_metrics": {
            "accuracy": mean_or_none(record["accuracy"] for record in scored),
            "exact_match": mean_or_none(record["exact_match"] for record in scored),
            "rouge1_f1": mean_or_none(record["rouge1_f1"] for record in scored),
            "bert_f1": bert_value,
            "bert_f1_status": bert_status,
            "final_score": final_score_value,
            "final_score_status": final_status,
            "sample_size": len(scored),
            "backend_em_agreement": mean_or_none(agreements),
            "backend_em_comparable": len(agreements),
        },
        "retrieval_metrics": {
            "recall_at_1": mean_or_none(record["retrieval"]["recall_at_1"] for record in retrieval_scored),
            "recall_at_3": mean_or_none(record["retrieval"]["recall_at_3"] for record in retrieval_scored),
            "recall_at_6": mean_or_none(record["retrieval"]["recall_at_6"] for record in retrieval_scored),
            "mrr": mean_or_none(record["retrieval"]["mrr"] for record in retrieval_scored),
            "sample_size": len(retrieval_scored),
            "skipped": len(succeeded) - len(retrieval_scored),
        },
        "operations": {
            "avg_duration_ms": mean_or_none(durations),
            "p95_duration_ms": percentile(durations, 0.95),
            "avg_chunk_count": mean_or_none(chunks),
            "avg_input_fragments": mean_or_none(fragments),
            "degraded_count": len(degraded),
            "degradation_rate": round(len(degraded) / len(succeeded), 6) if succeeded else None,
            "retrieval_protocol_distribution": dict(retrieval_protocols),
            "rerank_protocol_distribution": dict(rerank_protocols),
            "mode_distribution": dict(modes),
        },
    }

# --------------------------------------------------------------------------- #
# 4. 报告层
# --------------------------------------------------------------------------- #
METRIC_SPECS: tuple[tuple[str, str, tuple[str, str], bool | None], ...] = (
    ("exact_match", "EM（规范化精确匹配）", ("answer_metrics", "exact_match"), True),
    ("rouge1_f1", "ROUGE-1 F1（jieba）", ("answer_metrics", "rouge1_f1"), True),
    ("bert_f1", "BERTScore F1", ("answer_metrics", "bert_f1"), True),
    ("final_score", "官方综合分", ("answer_metrics", "final_score"), True),
    ("recall_at_1", "Recall@1", ("retrieval_metrics", "recall_at_1"), True),
    ("recall_at_3", "Recall@3", ("retrieval_metrics", "recall_at_3"), True),
    ("recall_at_6", "Recall@6", ("retrieval_metrics", "recall_at_6"), True),
    ("mrr", "MRR", ("retrieval_metrics", "mrr"), True),
    ("avg_duration_ms", "平均耗时(ms)", ("operations", "avg_duration_ms"), False),
    ("p95_duration_ms", "P95 耗时(ms)", ("operations", "p95_duration_ms"), False),
    ("avg_chunk_count", "平均 chunk 数", ("operations", "avg_chunk_count"), None),
    ("avg_input_fragments", "平均输入片段数", ("operations", "avg_input_fragments"), None),
    ("failure_rate", "失败率", ("totals", "failure_rate"), False),
    ("degradation_rate", "降级率", ("operations", "degradation_rate"), False),
)


def dig(payload: dict[str, Any], path: Sequence[str]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def build_comparison(arm_payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """两臂指标差异（第一臂为对照臂，第二臂为实验臂）。"""

    names = list(arm_payloads)
    arm_a = names[0]
    arm_b = names[1] if len(names) > 1 else None
    metrics: list[dict[str, Any]] = []
    for key, label, path, higher_is_better in METRIC_SPECS:
        value_a = dig(arm_payloads[arm_a]["aggregate"], path)
        value_b = dig(arm_payloads[arm_b]["aggregate"], path) if arm_b else None
        delta = None
        if isinstance(value_a, (int, float)) and isinstance(value_b, (int, float)):
            delta = round(value_b - value_a, 6)
        metrics.append(
            {
                "key": key,
                "label": label,
                "higher_is_better": higher_is_better,
                arm_a: value_a,
                arm_b: value_b,
                "delta": delta,
            }
        )
    per_question = []
    if arm_b:
        ids = [record["id"] for record in arm_payloads[arm_a]["questions"]]
        index_a = {record["id"]: record for record in arm_payloads[arm_a]["questions"]}
        index_b = {record["id"]: record for record in arm_payloads[arm_b]["questions"]}
        for question_id in ids:
            record_a = index_a[question_id]
            record_b = index_b.get(question_id, {})
            per_question.append(
                {
                    "id": question_id,
                    "document": record_a["document"],
                    "category_label": record_a["category_label"],
                    "target_page": record_a["target_page"],
                    "arms": {
                        arm_a: _question_view(record_a),
                        arm_b: _question_view(record_b),
                    },
                }
            )
    return {
        "baseline_arm": arm_a,
        "compare_arm": arm_b,
        "metrics": metrics,
        "per_question": per_question,
    }


def _question_view(record: dict[str, Any]) -> dict[str, Any]:
    retrieval = record.get("retrieval") or {}
    return {
        "ok": record.get("ok"),
        "prediction": record.get("prediction"),
        "exact_match": record.get("exact_match"),
        "rouge1_f1": record.get("rouge1_f1"),
        "bert_f1": record.get("bert_f1"),
        "final_score": record.get("final_score"),
        "duration_ms": record.get("duration_ms"),
        "first_hit_rank": retrieval.get("first_hit_rank"),
        "mrr": retrieval.get("mrr"),
        "error": record.get("error"),
    }


def build_report(
    selection: Selection,
    arm_payloads: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """汇总机器可读报告。"""

    failures: list[dict[str, Any]] = []
    degradations: list[dict[str, Any]] = []
    for arm, payload in arm_payloads.items():
        for record in payload["questions"]:
            if not record["ok"]:
                failures.append(
                    {
                        "arm": arm,
                        "id": record["id"],
                        "document": record["document"],
                        "attempts": record["attempts"],
                        "error": record["error"],
                    }
                )
            rag = record.get("rag") or {}
            if rag.get("degraded"):
                degradations.append(
                    {
                        "arm": arm,
                        "id": record["id"],
                        "document": record["document"],
                        "retrieval_protocol": rag.get("retrieval_protocol"),
                        "rerank_protocol": rag.get("rerank_protocol"),
                        "reason": rag.get("degradation_reason"),
                    }
                )
    report = {
        "tool": "evaluate_regression",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "selection": {
            "stage": config["stage"],
            "split": config["split"],
            "requested_size": selection.requested_size,
            "selected_count": len(selection.items),
            "seed": selection.seed,
            "available_total": selection.available_total,
            "available_documents": selection.available_documents,
            "excluded_documents": selection.excluded_documents,
            "document_count": len(selection.groups),
            "documents": selection.documents,
            "groups": selection.groups,
            "question_ids": [item.id for item in selection.items],
            "category_distribution": selection_distribution(selection, "primary"),
            "pattern_match_distribution": selection_distribution(selection, "pattern"),
            "target_page_available": sum(1 for item in selection.items if item.has_target_page),
            "target_page_missing": sum(1 for item in selection.items if not item.has_target_page),
            "with_reference": sum(1 for item in selection.items if item.has_reference),
        },
        "arms": arm_payloads,
        "comparison": build_comparison(arm_payloads),
        "failures": failures,
        "degradations": degradations,
    }
    report["conclusion"] = build_conclusion(report)
    return report


def build_conclusion(report: dict[str, Any]) -> list[str]:
    """基于真实计算结果生成结论；缺指标就写明未计算，不做推断。"""

    comparison = report["comparison"]
    metrics = {item["key"]: item for item in comparison["metrics"]}
    arm_a = comparison["baseline_arm"]
    arm_b = comparison["compare_arm"]
    lines: list[str] = []

    def describe(key: str, unit: str = "") -> str:
        item = metrics.get(key, {})
        value_a = item.get(arm_a)
        value_b = item.get(arm_b)
        if value_a is None and value_b is None:
            return f"{item.get('label', key)}：未计算"
        delta = item.get("delta")
        if delta is None:
            return f"{item.get('label', key)}：{arm_a}={value_a}，{arm_b}={value_b}"
        sign = "+" if delta >= 0 else ""
        return f"{item.get('label', key)}：{arm_a}={value_a}{unit}，{arm_b}={value_b}{unit}，差值 {sign}{delta}{unit}"

    if arm_b:
        lines.append(describe("exact_match"))
        lines.append(describe("rouge1_f1"))
        lines.append(describe("final_score"))
        lines.append(describe("failure_rate"))
        lines.append(describe("degradation_rate"))
    else:
        lines.append(f"仅执行单个评测臂 {arm_a}，未做对照。")

    final_status = dig(report["arms"][arm_a]["aggregate"], ("answer_metrics", "final_score_status"))
    if final_status and final_status != "已计算":
        lines.append(f"官方综合分{final_status}；本次结论仅基于 EM 与 ROUGE-1 F1。")

    rag_arm = "rag" if "rag" in report["arms"] else None
    if rag_arm:
        retrieval = report["arms"][rag_arm]["aggregate"]["retrieval_metrics"]
        lines.append(
            f"检索质量（rag 臂）：Recall@1={retrieval['recall_at_1']}，"
            f"Recall@3={retrieval['recall_at_3']}，Recall@6={retrieval['recall_at_6']}，"
            f"MRR={retrieval['mrr']}（样本 {retrieval['sample_size']} 题，"
            f"跳过 {retrieval['skipped']} 题无目标页提示）。"
        )
    return lines


def escape_cell(value: Any) -> str:
    if value is None:
        return "—"
    text = str(value)
    return text.replace("|", "\\|")


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(escape_cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(escape_cell(cell) for cell in row) + " |")
    return "\n".join(lines)


def format_metric(value: Any) -> str:
    """比例类指标（0-1）保留 4 位小数，耗时/数量类保留 2 位。"""

    if value is None:
        return "未计算"
    if isinstance(value, float):
        return f"{value:.4f}" if 0 <= value <= 1 else f"{value:.2f}"
    return str(value)


def format_delta(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:+.4f}" if 0 <= abs(value) <= 1 else f"{value:+.2f}"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    """生成人读报告。"""

    config = report["config"]
    selection = report["selection"]
    arms = report["arms"]
    comparison = report["comparison"]
    arm_a = comparison["baseline_arm"]
    arm_b = comparison["compare_arm"]
    lines: list[str] = []
    add = lines.append

    add("# 复赛回归评测与 A/B 对照报告")
    add("")
    add(f"- 生成时间：{report['generated_at']}")
    add(f"- 阶段 / 分区：{config['stage']} / {config['split']}")
    add(
        f"- 回归集：{selection['selected_count']} 题，来自 {selection['document_count']} 个 PDF"
        f"（候选池 {selection['available_total']} 题 / {selection['available_documents']} 个 PDF）"
    )
    add(f"- 随机种子：{selection['seed']}（同参数可完全复现）")
    add(
        f"- 服务地址：{config['base_url']} · 运行模式："
        f"{'演示模式 force_demo=true' if config['force_demo'] else '真实模型链路 force_demo=false'}"
    )
    add(f"- 评测臂：{' vs '.join(arms)}")
    if selection["excluded_documents"]:
        add(f"- 排除文档：{', '.join(selection['excluded_documents'])}")
    add("")

    add("## 1. 选集与分组")
    add("")
    add("### 1.1 题型分布（主分类，每题唯一归属）")
    add("")
    add(
        markdown_table(
            ["题型", "题量", "占比"],
            [
                [
                    CATEGORY_LABELS.get(name, name),
                    stats["count"],
                    f"{stats['ratio'] * 100:.1f}%",
                ]
                for name, stats in selection["category_distribution"].items()
            ],
        )
    )
    add("")
    add("### 1.2 PDF 分组（同一 PDF 的题连续排布，按位置切分不会拆散同一 PDF）")
    add("")
    add(
        markdown_table(
            ["#", "PDF", "题量", "题号区间"],
            [
                [index + 1, group["document"], group["count"], f"{group['start_index'] + 1}-{group['end_index'] + 1}"]
                for index, group in enumerate(selection["groups"])
            ],
        )
    )
    add("")
    add(
        f"- 可判定目标页的题目：{selection['target_page_available']} 题；"
        f"无「第 N 页」提示（不计入检索指标）：{selection['target_page_missing']} 题。"
    )
    add(f"- 题号清单：{', '.join(selection['question_ids'])}")
    add("")

    add("## 2. A/B 对照总表")
    add("")
    headers = ["指标", f"{arm_a}", f"{arm_b or '—'}"]
    if arm_b:
        headers.append("差值(第二臂-第一臂)")
    rows = []
    for item in comparison["metrics"]:
        row = [item["label"], format_metric(item.get(arm_a)), format_metric(item.get(arm_b)) if arm_b else "—"]
        if arm_b:
            row.append(format_delta(item.get("delta")))
        rows.append(row)

    add(markdown_table(headers, rows))
    add("")
    for arm, payload in arms.items():
        status = payload["aggregate"]["answer_metrics"]
        add(
            f"- {arm}：BERTScore {status['bert_f1_status']}；官方综合分 {status['final_score_status']}；"
            f"可评分样本 {status['sample_size']} 题。"
        )
    add("")

    add("## 3. 检索质量（Recall@K / MRR，目标页取题干「第 N 页」提示）")
    add("")
    rows = []
    for arm, payload in arms.items():
        retrieval = payload["aggregate"]["retrieval_metrics"]
        rows.append(
            [
                arm,
                format_metric(retrieval["recall_at_1"]),
                format_metric(retrieval["recall_at_3"]),
                format_metric(retrieval["recall_at_6"]),
                format_metric(retrieval["mrr"]),
                retrieval["sample_size"],
                retrieval["skipped"],
            ]
        )
    add(markdown_table(["评测臂", "Recall@1", "Recall@3", "Recall@6", "MRR", "样本数", "跳过数"], rows))
    add("")

    add("## 4. 运营指标")
    add("")
    rows = []
    for arm, payload in arms.items():
        operations = payload["aggregate"]["operations"]
        totals = payload["aggregate"]["totals"]
        rows.append(
            [
                arm,
                format_metric(operations["avg_duration_ms"]),
                format_metric(operations["p95_duration_ms"]),
                format_metric(operations["avg_chunk_count"]),
                format_metric(operations["avg_input_fragments"]),
                f"{totals['failed']}/{totals['questions']}",
                format_metric(totals["failure_rate"]),
                format_metric(operations["degradation_rate"]),
            ]
        )
    add(
        markdown_table(
            [
                "评测臂",
                "平均耗时(ms)",
                "P95(ms)",
                "平均 chunk 数",
                "平均输入片段数",
                "失败/总数",
                "失败率",
                "降级率",
            ],
            rows,
        )
    )
    add("")
    for arm, payload in arms.items():
        operations = payload["aggregate"]["operations"]
        add(
            f"- {arm} 召回协议分布：{operations['retrieval_protocol_distribution'] or '无'}；"
            f"重排协议分布：{operations['rerank_protocol_distribution'] or '无'}；"
            f"运行模式分布：{operations['mode_distribution'] or '无'}。"
        )
    add("")

    add("## 5. 失败清单")
    add("")
    if report["failures"]:
        add(
            markdown_table(
                ["评测臂", "题号", "PDF", "尝试次数", "错误码", "错误信息", "可重试"],
                [
                    [
                        failure["arm"],
                        failure["id"],
                        failure["document"],
                        failure["attempts"],
                        (failure["error"] or {}).get("code"),
                        (failure["error"] or {}).get("message"),
                        (failure["error"] or {}).get("retryable"),
                    ]
                    for failure in report["failures"]
                ],
            )
        )
    else:
        add("无失败题目。")
    add("")

    add("## 6. 降级清单")
    add("")
    if report["degradations"]:
        add(
            markdown_table(
                ["评测臂", "题号", "PDF", "召回协议", "重排协议", "降级原因"],
                [
                    [
                        item["arm"],
                        item["id"],
                        item["document"],
                        item["retrieval_protocol"],
                        item["rerank_protocol"],
                        item["reason"],
                    ]
                    for item in report["degradations"]
                ],
            )
        )
    else:
        add("无降级情况。")
    add("")

    add("## 7. 逐题明细")
    add("")
    if comparison["per_question"]:
        headers = ["题号", "PDF", "目标页", "题型"]
        for arm in arms:
            headers.extend([f"{arm} 答案", f"{arm} EM", f"{arm} ROUGE-1", f"{arm} 命中位次", f"{arm} 耗时(ms)"])
        rows = []
        for item in comparison["per_question"]:
            row = [item["id"], item["document"], item["target_page"], item["category_label"]]
            for arm in arms:
                view = item["arms"].get(arm) or {}
                row.extend(
                    [
                        view.get("prediction") or "（失败）",
                        format_metric(view.get("exact_match")),
                        format_metric(view.get("rouge1_f1")),
                        "—" if view.get("first_hit_rank") is None else view.get("first_hit_rank"),
                        "—" if view.get("duration_ms") is None else view.get("duration_ms"),
                    ]
                )
            rows.append(row)
        add(markdown_table(headers, rows))
    else:
        add("无逐题明细（仅执行单臂时仍见各臂明细表）。")
    add("")

    add("## 8. 结论")
    add("")
    for line in report["conclusion"]:
        add(f"- {line}")
    add("")
    add(
        "> 口径说明：EM 与 ROUGE-1 F1 直接复用后端 `app.services.evaluation` 的规范化实现"
        "（NFKC + casefold 严格相等 / rouge-score + jieba 分词），不存在第二套规范化逻辑；"
        "综合分遵循官方权重 0.5×BERTScore + 0.25×ROUGE-1 + 0.25×EM，缺 BERTScore 时标记为不完整而非补零。"
    )
    add("")
    return "\n".join(lines)


def create_output_dir(root: Path, stage: str, split: str, seed: int, stamp: str | None = None) -> Path:
    """创建不覆盖既有报告的输出目录。"""

    root.mkdir(parents=True, exist_ok=True)
    label = stamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    base = root / f"{stage}_{split}_{label}_{seed}"
    candidate = base
    index = 2
    while candidate.exists():
        candidate = root / f"{base.name}_{index}"
        index += 1
    candidate.mkdir(parents=True)
    return candidate


def write_report(output_dir: Path, report: dict[str, Any]) -> tuple[Path, Path]:
    json_path = output_dir / "report.json"
    md_path = output_dir / "report.md"
    atomic_write_text(json_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    atomic_write_text(md_path, render_markdown(report))
    return json_path, md_path

# --------------------------------------------------------------------------- #
# 5. 控制台输出与命令行入口
# --------------------------------------------------------------------------- #
def display_width(text: str) -> int:
    """中文按双宽度计算，保证控制台表格对齐。"""

    return sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in text)


def pad(text: str, width: int) -> str:
    return text + " " * max(0, width - display_width(text))


def render_console_table(report: dict[str, Any]) -> str:
    comparison = report["comparison"]
    arm_a = comparison["baseline_arm"]
    arm_b = comparison["compare_arm"]
    headers = ["指标", arm_a, arm_b or "—", "差值"]
    rows: list[list[str]] = []
    for item in comparison["metrics"]:
        rows.append(
            [
                item["label"],
                format_metric(item.get(arm_a)),
                format_metric(item.get(arm_b)) if arm_b else "—",
                format_delta(item.get("delta")),
            ]
        )
    widths = [
        max(
            [display_width(str(headers[index]))]
            + [display_width(str(row[index])) for row in rows]
        )
        for index in range(len(headers))
    ]
    lines = ["  ".join(pad(str(header), widths[index]) for index, header in enumerate(headers)).rstrip()]
    lines.append("  ".join("-" * widths[index] for index in range(len(headers))))
    for row in rows:
        lines.append("  ".join(pad(str(cell), widths[index]) for index, cell in enumerate(row)).rstrip())
    return "\n".join(lines)


def format_progress(arm: str, index: int, total: int, outcome: ArmOutcome) -> str:
    if outcome.ok:
        state = "成功"
    else:
        code = (outcome.error or {}).get("code", "UNKNOWN")
        state = f"失败[{code}]"
    duration = f"{outcome.duration_ms:.0f}ms" if outcome.duration_ms is not None else "—"
    return (
        f"[{arm}] {index + 1}/{total} id={outcome.question_id} doc={outcome.document} "
        f"{state} 耗时={duration} 尝试={outcome.attempts}"
    )


def render_selection_plan(selection: Selection, config: dict[str, Any]) -> str:
    lines: list[str] = []
    add = lines.append
    add(
        f"=== 回归集计划（{config['stage']}/{config['split']} · "
        f"{'dry-run：不调用模型、不写文件' if config['dry_run'] else '正式评测'}） ==="
    )
    add(
        f"题量：请求 {selection.requested_size} / 选中 {len(selection.items)}"
        f"（候选池 {selection.available_total} 题 · {selection.available_documents} 个 PDF）"
    )
    add(f"随机种子：{selection.seed}")
    add(f"排除文档：{', '.join(selection.excluded_documents) if selection.excluded_documents else '无'}")
    add(f"命中 PDF（{len(selection.groups)} 个）：")
    for group in selection.groups:
        add(f"  - {group['document']} × {group['count']}")
    distribution = selection_distribution(selection, "primary")
    add(
        "题型分布（主分类）："
        + " · ".join(
            f"{CATEGORY_LABELS.get(name, name)} {stats['count']}（{stats['ratio'] * 100:.1f}%）"
            for name, stats in distribution.items()
        )
    )
    add(
        f"检索可用性：可判定目标页 {sum(1 for item in selection.items if item.has_target_page)} 题 / "
        f"无页码提示 {sum(1 for item in selection.items if not item.has_target_page)} 题"
    )
    add("题目清单：")
    for item in selection.items:
        add(
            f"  [{item.id}] doc={item.document} 类别={CATEGORY_LABELS.get(item.category, item.category)}"
            f" 目标页={item.target_page if item.target_page else '—'}"
            f" 参考答案={(item.answer or '—')[:40]}"
        )
    return "\n".join(lines)


def flatten_excluded(values: Sequence[Sequence[str]]) -> list[str]:
    result: list[str] = []
    for group in values:
        for value in group:
            for part in str(value).split(","):
                text = part.strip()
                if text:
                    result.append(text)
    return result


def parse_arms(value: str) -> list[str]:
    arms = [part.strip() for part in str(value).split(",") if part.strip()]
    unknown = [arm for arm in arms if arm not in ARM_USE_RAG]
    if unknown:
        raise ValueError(f"未知评测臂：{', '.join(unknown)}（可选：baseline, rag）")
    if not arms:
        raise ValueError("--arms 不能为空")
    return arms


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate_regression.py",
        description=(
            "复赛回归评测与 A/B 对照工具：固定回归集（按 PDF 分组）-> 两臂执行 -> "
            "官方指标（EM / jieba ROUGE-1 F1 / 可选 BERTScore）+ 检索质量（Recall@K / MRR）-> 报告"
        ),
        epilog=(
            "示例：\n"
            "  python scripts/evaluate_regression.py --dry-run --size 8 --seed 20261025\n"
            "  python scripts/evaluate_regression.py --size 8 --seed 20261025 --arms baseline,rag\n"
            "  python scripts/evaluate_regression.py --size 12 --seed 20261025 "
            "--exclude-documents CN100342976C.pdf\n"
            "\n"
            "评测臂：baseline=目标页 OCR + VLM（use_rag=false）；"
            "rag=LangChain 分块 -> FAISS 召回 -> Reranker 重排 -> VLM（use_rag=true）。\n"
            "题目目录来自后端同一份 DatasetRepository 索引，--dry-run 不发起任何 HTTP 请求。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="后端服务地址（默认 %(default)s）")
    parser.add_argument("--stage", default="semi_final", choices=("semi_final", "preliminary"), help="阶段（默认 %(default)s）")
    parser.add_argument("--split", default="train", choices=("train", "test"), help="数据分区（默认 %(default)s）")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE, help="回归集题量（默认 %(default)s）")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="随机种子，决定选集结果（默认 %(default)s）")
    parser.add_argument("--arms", default="baseline,rag", help="评测臂，逗号分隔（默认 %(default)s）")
    parser.add_argument("--force-demo", action="store_true", help="强制演示模式（默认关闭，走真实模型链路）")
    parser.add_argument("--concurrency", type=int, default=1, help="单臂并发数（默认 %(default)s，上限 4）")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS, help="单次 HTTP 超时秒数（默认 %(default)s）")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES, help="可重试错误的最大重试次数（默认 %(default)s）")
    parser.add_argument("--backoff", type=float, default=DEFAULT_BACKOFF_SECONDS, help="重试退避基数秒数（默认 %(default)s）")
    parser.add_argument(
        "--exclude-documents",
        nargs="+",
        action="append",
        default=[],
        metavar="PDF",
        help="排除的 PDF 名称（可重复、可多值、逗号分隔），用于与其它实验做互斥分组",
    )
    parser.add_argument("--bertscore", action="store_true", help="启用可选的 BERTScore F1（在独立子进程中计算，需要 bert_score 依赖，未安装则报告「未计算」）")
    parser.add_argument("--bertscore-timeout", type=float, default=DEFAULT_BERTSCORE_TIMEOUT, help="BERTScore 子进程超时秒数（默认 %(default)s）")
    parser.add_argument("--bertscore-offline", action="store_true", help="BERTScore 子进程只用本地 HuggingFace 缓存（hub 不可达时使用）")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT), help="报告输出根目录（默认 %(default)s）")
    parser.add_argument("--dry-run", action="store_true", help="只做选集与计划输出：不调用模型、不写文件")
    return parser


def run_cli(args: argparse.Namespace) -> int:
    excluded = flatten_excluded(args.exclude_documents)
    arms = parse_arms(args.arms)
    concurrency = max(1, min(int(args.concurrency), MAX_CONCURRENCY))
    if int(args.concurrency) > MAX_CONCURRENCY:
        print(f"提示：--concurrency {args.concurrency} 超过上限 {MAX_CONCURRENCY}，已收敛为 {MAX_CONCURRENCY}")

    config: dict[str, Any] = {
        "stage": args.stage,
        "split": args.split,
        "size": args.size,
        "seed": args.seed,
        "arms": arms,
        "force_demo": bool(args.force_demo),
        "base_url": args.base_url,
        "concurrency": concurrency,
        "timeout": args.timeout,
        "max_retries": args.max_retries,
        "backoff": args.backoff,
        "exclude_documents": excluded,
        "bertscore_enabled": bool(args.bertscore),
        "bertscore_timeout": float(args.bertscore_timeout),
        "bertscore_offline": bool(args.bertscore_offline),
        "dry_run": bool(args.dry_run),
        "output_root": str(args.output_dir),
    }

    catalog = load_catalog(args.stage, args.split)
    selection = select_regression_set(catalog, args.size, args.seed, excluded)
    if not selection.items:
        raise ValueError("选集为空：请检查 --size / --exclude-documents 是否把题目全部排除了")
    print(render_selection_plan(selection, config))
    if args.dry_run:
        print("\ndry-run 结束：未调用任何模型，未写入任何文件。")
        return 0

    items_by_id = {item.id: item for item in selection.items}
    bert = BertScoreRunner(bool(args.bertscore))
    bert.timeout = float(args.bertscore_timeout)
    bert.offline = bool(args.bertscore_offline)
    arm_payloads: dict[str, dict[str, Any]] = {}

    with HttpApiClient(args.base_url, timeout=args.timeout) as client:
        for arm in arms:
            print(f"\n=== 开始评测臂 {ARM_LABELS[arm]} ===")
            outcomes = run_arm(
                client,
                arm,
                selection.items,
                stage=args.stage,
                split=args.split,
                force_demo=bool(args.force_demo),
                concurrency=concurrency,
                max_retries=args.max_retries,
                backoff_seconds=args.backoff,
                on_progress=lambda name, index, total, outcome: print(
                    format_progress(name, index, total, outcome), flush=True
                ),
            )
            pairs = [
                (outcome.question_id, outcome.answer or "", items_by_id[outcome.question_id].answer or "")
                for outcome in outcomes
                if outcome.ok and items_by_id[outcome.question_id].answer
            ]
            bert_scores = bert.score(pairs)
            records = build_question_records(args.stage, outcomes, items_by_id, bert_scores)
            aggregate = aggregate_arm(records, bert.status)
            arm_payloads[arm] = {
                "label": ARM_LABELS[arm],
                "use_rag": ARM_USE_RAG[arm],
                "aggregate": aggregate,
                "questions": records,
            }

    report = build_report(selection, arm_payloads, config)
    output_dir = create_output_dir(Path(args.output_dir), args.stage, args.split, args.seed)
    json_path, md_path = write_report(output_dir, report)

    print("\n=== A/B 对照结果 ===")
    print(render_console_table(report))
    print("\n结论：")
    for line in report["conclusion"]:
        print(f"  - {line}")
    print(f"\n报告已写入：\n  {json_path}\n  {md_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    force_utf8_output()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run_cli(args)
    except ValueError as exc:
        print(f"参数错误：{exc}", file=sys.stderr)
        return 2
    except AppError as exc:
        print(f"后端错误：[{exc.code}] {exc.message}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())