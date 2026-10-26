"""scripts/evaluate_regression.py 的离线单元测试。

约束：不访问网络、不调用真实模型。
- 选集 / 分类 / 指标 / 报告：纯函数与本地文件。
- 重试与失败：使用 httpx.MockTransport 模拟 500(retryable) 与 401(不可重试)。
- dry-run：用禁止实例化的假客户端断言零 HTTP 调用。
"""

from __future__ import annotations

import ast
import importlib.util
import io
import json
import sys
from pathlib import Path

import httpx
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts import bertscore_worker as worker_module  # noqa: E402
from scripts import evaluate_regression as er  # noqa: E402


CATEGORY_SAMPLES = {
    "spatial_relation": "在文件中第3页的示意图中，部件4相对于部件5的位置关系是？",
    "connection_state": "部件4与部件5的连接方式是怎样的？",
    "function_problem": "该装置的作用是什么？",
    "process_causal": "请按顺序说明第5页的工艺步骤。",
}


def make_item(
    question_id: str,
    document: str,
    question: str,
    answer: str | None = "参考答案",
    page: int | None = 3,
) -> er.QuestionItem:
    return er.QuestionItem(
        id=str(question_id),
        question=question,
        document=document,
        answer=answer,
        page_hint=page,
        has_reference=answer is not None,
        category=er.classify_question(question),
        target_page=er.parse_target_page(question),
    )


def build_catalog(documents: int = 4, per_document: int = 6) -> list[er.QuestionItem]:
    """构造一个多 PDF、多题型的目录（不依赖真实数据集）。"""

    items: list[er.QuestionItem] = []
    counter = 1
    for document_index in range(documents):
        document = f"CN{1000 + document_index}A.pdf"
        for slot in range(per_document):
            category = er.CATEGORY_ORDER[slot % len(er.CATEGORY_ORDER)]
            question = f"{CATEGORY_SAMPLES[category]}（样本{slot}）"
            items.append(make_item(counter, document, question, f"参考答案{slot}", 3 + slot))
            counter += 1
    return items


def make_outcome(question_id: str, document: str = "CN1000A.pdf", answer: str = "参考答案0") -> er.ArmOutcome:
    return er.ArmOutcome(
        question_id=question_id,
        document=document,
        ok=True,
        attempts=1,
        answer=answer,
        duration_ms=100.0,
        mode="real",
        rag={
            "enabled": True,
            "requested": True,
            "chunk_count": 42,
            "items": [{"page": 3}, {"page": 5}, {"page": 9}],
            "retrieval_protocol": "hybrid_keyword_vector_npy",
            "rerank_protocol": "compatible_rerank",
        },
        evidence_pages=[3],
    )


# --------------------------------------------------------------------------- #
# 1. 选集确定性
# --------------------------------------------------------------------------- #
def test_same_seed_and_size_produce_identical_id_sequence() -> None:
    catalog = build_catalog()

    first = er.select_regression_set(catalog, size=8, seed=42)
    second = er.select_regression_set(catalog, size=8, seed=42)

    assert [item.id for item in first.items] == [item.id for item in second.items]
    assert first.documents == second.documents
    assert len(first.items) == 8


def test_different_seed_can_produce_a_different_selection() -> None:
    catalog = build_catalog(documents=12, per_document=6)

    first = er.select_regression_set(catalog, size=8, seed=1)
    second = er.select_regression_set(catalog, size=8, seed=2)

    assert [item.id for item in first.items] != [item.id for item in second.items]


def test_selection_is_reproducible_after_a_fresh_catalog_build() -> None:
    catalog = build_catalog()

    first = er.select_regression_set(catalog, size=10, seed=20261025)
    second = er.select_regression_set(build_catalog(), size=10, seed=20261025)

    assert [item.id for item in first.items] == [item.id for item in second.items]


# --------------------------------------------------------------------------- #
# 2. PDF 分组与互斥
# --------------------------------------------------------------------------- #
def test_excluded_documents_never_appear_in_the_selection() -> None:
    catalog = build_catalog(documents=6, per_document=4)
    excluded = ["CN1002A.pdf", "CN1004A.pdf"]

    selection = er.select_regression_set(catalog, size=8, seed=7, exclude_documents=excluded)

    excluded_keys = {er.normalize_document_key(value) for value in excluded}
    selected_keys = {er.normalize_document_key(value) for value in selection.documents}
    assert selected_keys.isdisjoint(excluded_keys)
    for item in selection.items:
        assert er.normalize_document_key(item.document) not in excluded_keys


def test_exclusion_accepts_name_without_pdf_suffix() -> None:
    catalog = build_catalog(documents=4, per_document=4)

    selection = er.select_regression_set(catalog, size=6, seed=3, exclude_documents=["CN1001A"])

    assert "CN1001A.pdf" not in selection.documents


def test_questions_of_the_same_document_stay_in_one_contiguous_block() -> None:
    catalog = build_catalog(documents=5, per_document=4)

    selection = er.select_regression_set(catalog, size=10, seed=11)

    assert len({group["document"] for group in selection.groups}) == len(selection.groups)
    for group in selection.groups:
        block = selection.items[group["start_index"] : group["end_index"] + 1]
        assert len(block) == group["count"]
        assert {item.document for item in block} == {group["document"]}
    # 不同 PDF 的块互不重叠，且拼回完整题集
    assert sum(group["count"] for group in selection.groups) == len(selection.items)


def test_selected_documents_are_disjoint_from_another_experiment_selection() -> None:
    catalog = build_catalog(documents=8, per_document=4)

    experiment_a = er.select_regression_set(catalog, size=8, seed=5)
    experiment_b = er.select_regression_set(catalog, size=8, seed=5, exclude_documents=experiment_a.documents)

    assert set(experiment_a.documents).isdisjoint(set(experiment_b.documents))


# --------------------------------------------------------------------------- #
# 3. 题型分类
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("sample", "expected"),
    [
        (CATEGORY_SAMPLES["spatial_relation"], "spatial_relation"),
        (CATEGORY_SAMPLES["connection_state"], "connection_state"),
        (CATEGORY_SAMPLES["function_problem"], "function_problem"),
        (CATEGORY_SAMPLES["process_causal"], "process_causal"),
    ],
)
def test_classification_hits_the_four_audit_categories(sample: str, expected: str) -> None:
    assert er.classify_question(sample) == expected
    assert expected in er.match_categories(sample)


def test_target_page_is_parsed_from_the_question_text() -> None:
    assert er.parse_target_page("在文件中第12页的示意图中，部件4位于部件5的左侧") == 12
    assert er.parse_target_page("该装置的作用是什么") is None


def test_selection_covers_all_four_categories_when_the_pool_allows() -> None:
    catalog = build_catalog(documents=4, per_document=8)

    selection = er.select_regression_set(catalog, size=8, seed=3)

    assert set(selection.primary_category_counts) >= set(er.CATEGORY_ORDER)
    assert all(selection.primary_category_counts[name] > 0 for name in er.CATEGORY_ORDER)


# --------------------------------------------------------------------------- #
# 4. 指标
# --------------------------------------------------------------------------- #
def test_em_and_rouge1_follow_the_backend_normalization() -> None:
    identical = er.answer_metrics("semi_final", "部件4位于部件5的左侧", "部件4位于部件5的左侧")
    assert identical["exact_match"] == 1.0
    assert identical["rouge1_f1"] == pytest.approx(1.0)

    # jieba 分词后 7 个 token 中命中 6 个 -> F1 = 6/7
    partial = er.answer_metrics("semi_final", "部件4位于部件5的右侧", "部件4位于部件5的左侧")
    assert partial["exact_match"] == 0.0
    assert partial["rouge1_f1"] == pytest.approx(6 / 7)

    unrelated = er.answer_metrics("semi_final", "完全无关", "用于快速组装定位")
    assert unrelated["exact_match"] == 0.0
    assert unrelated["rouge1_f1"] == 0.0


def test_metrics_are_not_applicable_without_a_reference() -> None:
    metrics = er.answer_metrics("semi_final", "任意答案", None)

    assert metrics["exact_match"] is None
    assert metrics["rouge1_f1"] is None
    assert metrics["final_score"] is None


def test_recall_and_mrr_use_the_rag_item_order() -> None:
    items = [{"page": 3}, {"page": 7}, {"page": 5}, {"page": 9}]

    metrics = er.retrieval_metrics(items, target_page=5)

    assert metrics["applicable"] is True
    assert metrics["first_hit_rank"] == 3
    assert metrics["mrr"] == pytest.approx(1 / 3)
    assert metrics["recall_at_1"] == 0.0
    assert metrics["recall_at_3"] == 1.0
    assert metrics["recall_at_6"] == 1.0


def test_recall_at_one_requires_the_target_page_in_the_first_slot() -> None:
    metrics = er.retrieval_metrics([{"page": 5}, {"page": 3}], target_page=3)

    assert metrics["first_hit_rank"] == 2
    assert metrics["recall_at_1"] == 0.0
    assert metrics["recall_at_3"] == 1.0


def test_retrieval_metrics_are_skipped_without_a_target_page() -> None:
    metrics = er.retrieval_metrics([{"page": 3}], target_page=None)

    assert metrics["applicable"] is False
    assert metrics["mrr"] is None
    assert metrics["recall_at_3"] is None


class _FakeCompleted:
    """subprocess.CompletedProcess 的最小替身。"""

    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _force_bertscore_available(monkeypatch) -> None:
    """让可用性探测通过，但不真的 import bert_score。"""

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "bert_score":
            return object()
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(er.importlib.util, "find_spec", fake_find_spec)


def test_bertscore_probe_never_imports_bert_score_or_torch(monkeypatch) -> None:
    """可用性探测只能用 find_spec，不能把 torch 拉进主进程。"""

    _force_bertscore_available(monkeypatch)
    watched = {"bert_score", "torch"}
    before = {name for name in sys.modules if name.split(".")[0] in watched}

    runner = er.BertScoreRunner(enabled=True)

    after = {name for name in sys.modules if name.split(".")[0] in watched}
    assert after == before, "可用性探测不得导入 bert_score / torch"
    assert runner.available is True
    assert runner.status == "待计算"


def test_bertscore_missing_dependency_is_reported_as_not_computed(monkeypatch) -> None:
    monkeypatch.setattr(er.importlib.util, "find_spec", lambda name, *args, **kwargs: None)

    runner = er.BertScoreRunner(enabled=True)

    assert runner.available is False
    assert "未计算" in runner.status and "未安装" in runner.status
    assert runner.score([("q1", "预测", "参考")]) == {}


def test_bertscore_is_computed_in_a_subprocess_and_parsed(monkeypatch) -> None:
    _force_bertscore_available(monkeypatch)
    calls: list[tuple[tuple[str, ...], dict]] = []

    def fake_run(command, **kwargs):
        calls.append((tuple(command), kwargs))
        return _FakeCompleted(
            stdout=json.dumps(
                {"f1": [0.8446, 1.0], "model_type": "bert-base-chinese", "lang": "zh"}
            )
        )

    monkeypatch.setattr(er.subprocess, "run", fake_run)
    runner = er.BertScoreRunner(enabled=True)

    scores = runner.score([("q1", "预测一", "参考一"), ("q2", "预测二", "参考二")])

    assert scores == {"q1": 0.8446, "q2": 1.0}
    assert len(calls) == 1, "成功时只应拉起一次子进程"
    command, kwargs = calls[0]
    assert command[0] == sys.executable
    assert command[1].endswith("bertscore_worker.py")
    assert json.loads(kwargs["input"])["pairs"] == [["预测一", "参考一"], ["预测二", "参考二"]]
    assert "已计算" in runner.status
    assert "独立子进程" in runner.status


def test_bertscore_subprocess_failure_is_reported_as_not_computed(monkeypatch) -> None:
    _force_bertscore_available(monkeypatch)
    monkeypatch.setattr(
        er.subprocess,
        "run",
        lambda command, **kwargs: _FakeCompleted(returncode=4, stderr="worker boom\n"),
    )
    runner = er.BertScoreRunner(enabled=True)

    scores = runner.score([("q1", "预测", "参考")])

    assert scores == {}
    assert runner.status.startswith("未计算")
    assert "worker boom" in runner.status


def test_bertscore_invalid_worker_output_is_reported_as_not_computed(monkeypatch) -> None:
    _force_bertscore_available(monkeypatch)
    monkeypatch.setattr(
        er.subprocess,
        "run",
        lambda command, **kwargs: _FakeCompleted(stdout="not json at all"),
    )
    runner = er.BertScoreRunner(enabled=True)

    scores = runner.score([("q1", "预测", "参考")])

    assert scores == {}
    assert "未计算" in runner.status
    assert "JSON" in runner.status


def test_bertscore_worker_failure_falls_back_to_cache_only_mode(monkeypatch) -> None:
    """hub 不可达时，失败后应再试一次「只用本地缓存」模式。"""

    _force_bertscore_available(monkeypatch)
    payloads: list[dict] = []

    def fake_run(command, **kwargs):
        payload = json.loads(kwargs["input"])
        payloads.append(payload)
        if not payload.get("offline"):
            return _FakeCompleted(returncode=1, stderr="ConnectTimeout\n")
        return _FakeCompleted(stdout=json.dumps({"f1": [0.75], "model_type": "bert-base-chinese"}))

    monkeypatch.setattr(er.subprocess, "run", fake_run)
    runner = er.BertScoreRunner(enabled=True)

    scores = runner.score([("q1", "预测", "参考")])

    assert scores == {"q1": 0.75}
    assert len(payloads) == 2
    assert payloads[0]["offline"] is False
    assert payloads[1]["offline"] is True
    assert "已计算" in runner.status


def test_bertscore_unavailable_is_reported_as_not_computed() -> None:
    """未开启 BERTScore 时，报告里必须写「未计算」而不是补 0。"""

    runner = er.BertScoreRunner(enabled=False)

    item = make_item("1", "CN1000A.pdf", CATEGORY_SAMPLES["spatial_relation"], "参考", 3)
    records = er.build_question_records(
        "semi_final",
        [make_outcome("1", answer="预测")],
        {"1": item},
        {},
    )
    aggregate = er.aggregate_arm(records, runner.status)

    assert aggregate["answer_metrics"]["bert_f1"] is None
    assert "未计算" in aggregate["answer_metrics"]["bert_f1_status"]
    assert aggregate["answer_metrics"]["final_score"] is None
    assert "不完整" in aggregate["answer_metrics"]["final_score_status"]


def test_bertscore_disabled_is_reported_as_not_computed() -> None:
    runner = er.BertScoreRunner(enabled=False)

    assert runner.available is False
    assert "未计算" in runner.status
    assert runner.score([("1", "预测", "参考")]) == {}


# --------------------------------------------------------------------------- #
# 4b. BERTScore 子进程契约（主进程不得导入 bert_score / torch）
# --------------------------------------------------------------------------- #
def _imported_top_level_modules(path: Path) -> set[str]:
    """解析源码里真正出现的 import（不把文档字符串里的字样算进去）。"""

    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_only_the_worker_module_may_import_bert_score_or_torch() -> None:
    """torch 与 faiss 同进程共存会导致原生崩溃，导入权必须收在 worker 手里。"""

    forbidden = {"bert_score", "torch"}
    evaluate_path = ROOT_DIR / "scripts" / "evaluate_regression.py"
    worker_path = ROOT_DIR / "scripts" / "bertscore_worker.py"

    assert _imported_top_level_modules(evaluate_path).isdisjoint(forbidden)
    assert _imported_top_level_modules(Path(__file__)).isdisjoint(forbidden)
    assert "bert_score" in _imported_top_level_modules(worker_path)


def test_evaluate_script_launches_the_worker_instead_of_importing_it() -> None:
    source = (ROOT_DIR / "scripts" / "evaluate_regression.py").read_text(encoding="utf-8")

    assert "bertscore_worker.py" in source
    assert "subprocess.run(" in source


def test_worker_module_validates_its_payload_without_importing_bert_score() -> None:
    watched = {"bert_score", "torch"}
    before = {name for name in sys.modules if name.split(".")[0] in watched}

    assert worker_module.read_pairs({"pairs": [["预测", "参考"]]}) == (["预测"], ["参考"])
    assert worker_module.read_pairs({"pairs": []}) is None
    assert worker_module.read_pairs({"pairs": [["只有一侧"]]}) is None
    assert worker_module.read_pairs({"pairs": "不是列表"}) is None
    assert worker_module.read_pairs(["不是对象"]) is None

    after = {name for name in sys.modules if name.split(".")[0] in watched}
    assert after == before, "worker 的载荷校验不得导入 bert_score / torch"


def test_worker_rejects_invalid_stdin_without_importing_bert_score(monkeypatch) -> None:
    watched = {"bert_score", "torch"}
    before = {name for name in sys.modules if name.split(".")[0] in watched}
    monkeypatch.setattr(sys, "stdin", io.StringIO("这不是 JSON"))

    exit_code = worker_module.main()

    after = {name for name in sys.modules if name.split(".")[0] in watched}
    assert exit_code == 2
    assert after == before


def test_worker_rejects_empty_pairs_without_importing_bert_score(monkeypatch) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"pairs": []})))

    assert worker_module.main() == 2


# --------------------------------------------------------------------------- #
# 5. 报告生成
# --------------------------------------------------------------------------- #
def build_report_fixture(tmp_path: Path) -> dict:
    catalog = build_catalog()
    selection = er.select_regression_set(catalog, size=8, seed=20261025)
    items_by_id = {item.id: item for item in selection.items}
    arm_payloads: dict[str, dict] = {}
    for arm in ("baseline", "rag"):
        outcomes = [
            make_outcome(item.id, document=item.document, answer=item.answer or "答案")
            for item in selection.items
        ]
        records = er.build_question_records("semi_final", outcomes, items_by_id, {})
        arm_payloads[arm] = {
            "label": er.ARM_LABELS[arm],
            "use_rag": er.ARM_USE_RAG[arm],
            "aggregate": er.aggregate_arm(records, "未计算（未开启 --bertscore）"),
            "questions": records,
        }
    config = {
        "stage": "semi_final",
        "split": "train",
        "size": 8,
        "seed": 20261025,
        "arms": ["baseline", "rag"],
        "force_demo": False,
        "base_url": "http://127.0.0.1:8000",
        "concurrency": 1,
        "timeout": 180.0,
        "max_retries": 2,
        "backoff": 1.5,
        "exclude_documents": [],
        "bertscore_enabled": False,
        "dry_run": False,
        "output_root": str(tmp_path),
    }
    report = er.build_report(selection, arm_payloads, config)
    return {"report": report, "selection": selection}


def test_report_files_are_generated_with_both_arms(tmp_path: Path) -> None:
    fixture = build_report_fixture(tmp_path)

    json_path, md_path = er.write_report(tmp_path, fixture["report"])

    assert json_path.exists() and md_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert set(payload["arms"]) == {"baseline", "rag"}
    assert payload["arms"]["baseline"]["use_rag"] is False
    assert payload["arms"]["rag"]["use_rag"] is True
    assert payload["comparison"]["metrics"]
    assert payload["selection"]["question_ids"]

    markdown = md_path.read_text(encoding="utf-8")
    assert "baseline" in markdown and "rag" in markdown
    assert "EM（规范化精确匹配）" in markdown
    assert payload["arms"]["baseline"]["aggregate"]["answer_metrics"]["final_score"] is None


def test_repeated_runs_do_not_overwrite_existing_reports(tmp_path: Path) -> None:
    first = er.create_output_dir(tmp_path, "semi_final", "train", 42, stamp="20260101-000000")
    (first / "report.json").write_text("{}", encoding="utf-8")

    second = er.create_output_dir(tmp_path, "semi_final", "train", 42, stamp="20260101-000000")

    assert first != second
    assert first.exists() and second.exists()
    assert (first / "report.json").read_text(encoding="utf-8") == "{}"


# --------------------------------------------------------------------------- #
# 6. 失败处理与重试
# --------------------------------------------------------------------------- #
def build_mock_transport(calls: list[str]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        question_id = str(payload["question_id"])
        calls.append(question_id)
        if question_id == "retry":
            return httpx.Response(
                500,
                json={
                    "error": {
                        "code": "INTERNAL_ERROR",
                        "message": "服务内部错误",
                        "request_id": "req-retry",
                        "retryable": True,
                    }
                },
            )
        if question_id == "auth":
            return httpx.Response(
                401,
                json={
                    "error": {
                        "code": "UNAUTHORIZED",
                        "message": "鉴权失败",
                        "request_id": "req-auth",
                        "retryable": False,
                    }
                },
            )
        return httpx.Response(
            200,
            json={
                "run_id": "run-1",
                "status": "completed",
                "result": {
                    "answer": "参考答案0",
                    "duration_ms": 42.0,
                    "mode": "real",
                    "evidence_pages": [3],
                    "rag": {
                        "enabled": True,
                        "requested": True,
                        "chunk_count": 12,
                        "items": [{"page": 3}],
                        "retrieval_protocol": "hybrid_keyword_vector_npy",
                        "rerank_protocol": "compatible_rerank",
                    },
                },
            },
        )

    return httpx.MockTransport(handler)


def test_retryable_error_is_retried_and_non_retryable_error_is_not() -> None:
    calls: list[str] = []
    client = er.HttpApiClient("http://127.0.0.1:8000", transport=build_mock_transport(calls))
    items = [
        make_item("retry", "CN1000A.pdf", CATEGORY_SAMPLES["spatial_relation"], "参考答案0", 3),
        make_item("auth", "CN1000A.pdf", CATEGORY_SAMPLES["function_problem"], "参考答案1", 3),
        make_item("ok", "CN1000A.pdf", CATEGORY_SAMPLES["process_causal"], "参考答案2", 3),
    ]

    outcomes = er.run_arm(
        client,
        "baseline",
        items,
        stage="semi_final",
        split="train",
        force_demo=False,
        max_retries=2,
        backoff_seconds=0.0,
        sleep=lambda _seconds: None,
    )
    client.close()

    assert calls.count("retry") == 3, "500(retryable=true) 应重试 2 次共 3 次调用"
    assert calls.count("auth") == 1, "401(retryable=false) 不应重试"
    assert calls.count("ok") == 1
    assert [outcome.ok for outcome in outcomes] == [False, False, True]
    assert outcomes[0].error is not None and outcomes[0].error["code"] == "INTERNAL_ERROR"
    assert outcomes[1].error is not None and outcomes[1].error["code"] == "UNAUTHORIZED"
    assert outcomes[1].error["retryable"] is False
    assert outcomes[0].attempts == 3
    assert outcomes[1].attempts == 1


def test_failed_questions_are_counted_and_do_not_break_the_batch() -> None:
    calls: list[str] = []
    client = er.HttpApiClient("http://127.0.0.1:8000", transport=build_mock_transport(calls))
    items = [
        make_item("retry", "CN1000A.pdf", CATEGORY_SAMPLES["spatial_relation"], "参考答案0", 3),
        make_item("ok", "CN1000A.pdf", CATEGORY_SAMPLES["function_problem"], "参考答案1", 3),
    ]

    outcomes = er.run_arm(
        client,
        "rag",
        items,
        stage="semi_final",
        split="train",
        force_demo=False,
        max_retries=1,
        backoff_seconds=0.0,
        sleep=lambda _seconds: None,
    )
    records = er.build_question_records(
        "semi_final", outcomes, {item.id: item for item in items}, {}
    )
    aggregate = er.aggregate_arm(records, "未计算（未开启 --bertscore）")
    client.close()

    assert aggregate["totals"]["questions"] == 2
    assert aggregate["totals"]["failed"] == 1
    assert aggregate["totals"]["succeeded"] == 1
    assert aggregate["totals"]["failure_rate"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# 7. dry-run 零 HTTP
# --------------------------------------------------------------------------- #
def test_dry_run_issues_no_http_request_and_writes_no_file(monkeypatch, tmp_path: Path) -> None:
    class ForbiddenClient:
        def __init__(self, *args, **kwargs):
            raise AssertionError("dry-run 不应创建 HTTP 客户端")

    monkeypatch.setattr(er, "HttpApiClient", ForbiddenClient)
    monkeypatch.setattr(er, "load_catalog", lambda stage, split: build_catalog())
    output_root = tmp_path / "regression"

    exit_code = er.main(
        [
            "--dry-run",
            "--size",
            "6",
            "--seed",
            "5",
            "--output-dir",
            str(output_root),
        ]
    )

    assert exit_code == 0
    assert not output_root.exists()


# --------------------------------------------------------------------------- #
# 8. 端到端（假客户端）：两臂题集一致 + 报告落盘
# --------------------------------------------------------------------------- #
def test_both_arms_run_the_same_question_set_and_write_a_report(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, bool]] = []

    class FakeClient:
        def __init__(self, base_url: str, timeout: float = 180.0, transport=None) -> None:
            self.base_url = base_url

        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *exc_info: object) -> None:
            return None

        def close(self) -> None:
            return None

        def analyze(self, *, stage, split, question_id, use_rag, force_demo, use_ocr=True):
            calls.append((str(question_id), bool(use_rag)))
            return {
                "run_id": "run-fake",
                "status": "completed",
                "result": {
                    "answer": "参考答案0",
                    "duration_ms": 30.0,
                    "mode": "real",
                    "evidence_pages": [3],
                    "rag": {
                        "enabled": bool(use_rag),
                        "requested": bool(use_rag),
                        "chunk_count": 8,
                        "items": [{"page": 3}],
                        "retrieval_protocol": "hybrid_keyword_vector_npy",
                        "rerank_protocol": "compatible_rerank",
                    },
                },
            }

    monkeypatch.setattr(er, "HttpApiClient", FakeClient)
    monkeypatch.setattr(er, "load_catalog", lambda stage, split: build_catalog())
    output_root = tmp_path / "regression"

    exit_code = er.main(
        [
            "--size",
            "6",
            "--seed",
            "20261025",
            "--output-dir",
            str(output_root),
        ]
    )

    assert exit_code == 0
    baseline_ids = [question_id for question_id, use_rag in calls if not use_rag]
    rag_ids = [question_id for question_id, use_rag in calls if use_rag]
    assert baseline_ids == rag_ids, "两臂必须跑同一题集"
    assert len(baseline_ids) == 6

    reports = list(output_root.glob("*/report.json"))
    assert len(reports) == 1
    payload = json.loads(reports[0].read_text(encoding="utf-8"))
    assert set(payload["arms"]) == {"baseline", "rag"}
    assert payload["arms"]["baseline"]["aggregate"]["totals"]["questions"] == 6
    assert (reports[0].parent / "report.md").exists()