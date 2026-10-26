import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import SettingsStore
from app.main import app
from app.models.schemas import AnalyzeRequest, QuestionRecord
from app.services import analysis as analysis_module
from app.services.analysis import AnalysisPipeline, RunStore
from app.errors import AppError
from app.services.openai_client import extract_message_content


def test_vlm_parser_returns_auditable_summary_without_exposing_raw_payload() -> None:
    payload = json.dumps(
        {
            "answer": "B",
            "rationale": "OCR 文本明确说明主要功能为靠边规整。",
            "analysis_summary": [
                "从 OCR 文本定位到主要功能描述。",
                "将描述与 B 选项进行语义对齐。",
            ],
        },
        ensure_ascii=False,
    )

    answer, rationale, summary = AnalysisPipeline._parse_vlm(payload, "preliminary")

    assert answer == "B"
    assert "靠边规整" in rationale
    assert summary == ["从 OCR 文本定位到主要功能描述。", "将描述与 B 选项进行语义对齐。"]


def test_vlm_parser_falls_back_to_rationale_as_summary() -> None:
    payload = json.dumps({"answer": "部件4位于部件5左侧", "rationale": "图中标号4在标号5左侧。"}, ensure_ascii=False)

    answer, rationale, summary = AnalysisPipeline._parse_vlm(payload, "semi_final")

    assert answer == "部件4位于部件5左侧"
    assert summary == [rationale]


def test_semifinal_parser_strips_answer_prefix_and_discards_option_analysis() -> None:
    payload = json.dumps(
        {
            "answer": "最终答案为：部件7、部件8",
            "rationale": "两个部件均为问题要求的完整答案。",
            "option_analysis": [
                {"option": "A", "verdict": "selected", "reason": "不应出现在开放问答结果中"}
            ],
        },
        ensure_ascii=False,
    )

    parsed = AnalysisPipeline._parse_vlm_details(payload, "semi_final", target_page=3)

    assert parsed["answer"] == "部件7、部件8"
    assert parsed["option_analysis"] == []


def test_vlm_details_preserve_auditable_explanation_evidence_and_option_analysis() -> None:
    payload = json.dumps(
        {
            "answer": "B",
            "rationale": "编号 4 对应规整挡板。",
            "answer_explanation": "目标图中编号 4 位于输送带侧边，其结构和位置与规整挡板一致。",
            "evidence": [
                {
                    "page": 3,
                    "source": "target_image",
                    "quote": "编号 4 位于输送带侧边",
                    "relevance": "支持 B 所述的规整功能",
                },
                {
                    "page": 2,
                    "source": "OCR_TEXT",
                    "quote": "挡板用于货物靠边规整",
                    "relevance": "文字说明与目标图结构相互印证",
                },
            ],
            "option_analysis": [
                {"option": "A", "verdict": "rejected", "reason": "图中没有加热组件。"},
                {"option": "B", "verdict": "supported", "reason": "图文均指向规整挡板。"},
                {"option": "C", "verdict": "exclude", "reason": "没有分拣执行机构。"},
                {"option": "D", "verdict": "uncertain", "reason": "未看到称重传感器。"},
            ],
        },
        ensure_ascii=False,
    )

    parsed = AnalysisPipeline._parse_vlm_details(payload, "preliminary", target_page=3)

    assert parsed["answer"] == "B"
    assert "结构和位置" in parsed["answer_explanation"]
    assert parsed["evidence"] == [
        {
            "page": 3,
            "source": "target_image",
            "quote": "编号 4 位于输送带侧边",
            "relevance": "支持 B 所述的规整功能",
        },
        {
            "page": 2,
            "source": "ocr",
            "quote": "挡板用于货物靠边规整",
            "relevance": "文字说明与目标图结构相互印证",
        },
    ]
    assert [item["option"] for item in parsed["option_analysis"]] == ["A", "B", "C", "D"]
    assert next(item for item in parsed["option_analysis"] if item["option"] == "B")["verdict"] == "selected"


def test_vlm_evidence_invalid_pages_fall_back_to_target_and_sources_are_normalized() -> None:
    payload = json.dumps(
        {
            "answer": "A",
            "evidence": [
                {"page": -1, "source": "ocr", "quote": "负页码", "relevance": "测试"},
                {"page": "not-a-page", "source": "recognized_text", "quote": "坏页码", "relevance": "测试"},
                {"page": 0, "source": "visual_observation", "quote": "零页码", "relevance": "测试"},
                {"source": "unknown-provider", "quote": "缺失页码", "relevance": "测试"},
            ],
        },
        ensure_ascii=False,
    )

    parsed = AnalysisPipeline._parse_vlm_details(payload, "preliminary", target_page=7)

    assert [item["page"] for item in parsed["evidence"]] == [7, 7, 7, 7]
    assert [item["source"] for item in parsed["evidence"]] == [
        "ocr",
        "ocr",
        "target_image",
        "target_image",
    ]


def test_vlm_preliminary_fills_missing_abcd_option_analysis() -> None:
    payload = json.dumps(
        {
            "answer": "C",
            "explanation": "结构图中的联动皮带与 C 一致。",
            "choice_analysis": {
                "A": {"status": "incorrect", "reason": "编号不符。"},
                "C": {"status": "correct", "reason": "编号与部件对应。"},
            },
        },
        ensure_ascii=False,
    )

    parsed = AnalysisPipeline._parse_vlm_details(payload, "preliminary", target_page=5)

    assert [item["option"] for item in parsed["option_analysis"]] == ["A", "B", "C", "D"]
    by_option = {item["option"]: item for item in parsed["option_analysis"]}
    assert by_option["A"]["verdict"] == "rejected"
    assert by_option["C"]["verdict"] == "selected"
    assert by_option["B"]["verdict"] == "uncertain"
    assert by_option["D"]["verdict"] == "uncertain"


def test_vlm_old_formats_remain_compatible() -> None:
    legacy_json = json.dumps(
        {"answer": "D", "reason": "旧版 reason 字段仍应作为依据。", "reasoning_summary": "旧版单条摘要。"},
        ensure_ascii=False,
    )
    answer, rationale, summary = AnalysisPipeline._parse_vlm(legacy_json, "preliminary")
    assert (answer, rationale, summary) == ("D", "旧版 reason 字段仍应作为依据。", ["旧版单条摘要。"])

    plain_answer, plain_rationale, plain_summary = AnalysisPipeline._parse_vlm("答案为 B。", "preliminary")
    assert plain_answer == "B"
    assert plain_rationale == "模型未提供额外依据。"
    assert plain_summary == [plain_rationale]


def test_vlm_parser_ignores_reasoning_template_letters_before_embedded_json() -> None:
    content = """<think>Compare A/B/C/D and follow the requested template.</think>
Here is the result:
```json
{"answer":"B","answer_explanation":"功能描述直接对应靠边规整。","evidence":[{"page":2,"source":"ocr","quote":"靠边规整","relevance":"支持 B"}],"option_analysis":[{"option":"A","verdict":"rejected","reason":"无加热"},{"option":"B","verdict":"selected","reason":"功能一致"},{"option":"C","verdict":"rejected","reason":"无分拣"},{"option":"D","verdict":"rejected","reason":"无称重"}]}
```
"""
    parsed = AnalysisPipeline._parse_vlm_details(content, "preliminary", target_page=5)
    assert parsed["answer"] == "B"
    assert parsed["output_format"] == "json"
    assert parsed["audit_complete"] is True


def test_openai_content_extractor_removes_reasoning_blocks() -> None:
    content = extract_message_content(
        {"choices": [{"message": {"content": '<think>private reasoning A/B</think>\n{"answer":"B"}'}}]}
    )
    assert content == '{"answer":"B"}'


def test_vlm_parser_rejects_unstructured_template_that_only_mentions_options() -> None:
    content = "I cannot see the question. Please provide options A, B, C and D."
    try:
        AnalysisPipeline._parse_vlm_details(content, "preliminary", target_page=1)
    except AppError as exc:
        assert exc.code == "MODEL_ANSWER_INVALID"
    else:
        raise AssertionError("template text must not be treated as answer A")


def test_vlm_plain_answer_line_is_accepted_but_marked_incomplete() -> None:
    parsed = AnalysisPipeline._parse_vlm_details("最终答案：C", "preliminary", target_page=5)
    assert parsed["answer"] == "C"
    assert parsed["output_format"] == "plain_text"
    assert parsed["audit_complete"] is False
    assert "模型返回了非结构化文本" in parsed["output_warnings"]


def test_demo_analysis_api_returns_auditable_explanation_fields() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/analyze",
            json={
                "stage": "preliminary",
                "split": "train",
                "question_id": 1,
                "force_demo": True,
            },
        )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["answer"] == "B"
    assert result["answer_explanation"]
    assert result["evidence"] == [
        {
            "page": None,
            "source": "dataset_reference",
            "quote": "数据集参考答案：B",
            "relevance": "仅用于演示回放和界面验证，不属于 OCR 或目标图证据。",
        }
    ]
    assert [item["option"] for item in result["option_analysis"]] == ["A", "B", "C", "D"]
    assert next(item for item in result["option_analysis"] if item["option"] == "B")["verdict"] == "selected"


class _RepositoryStub:
    def __init__(self) -> None:
        self.question = QuestionRecord(
            stage="preliminary",
            split="train",
            id=7,
            question="第3页设计图中的机构用于什么？",
            document="sample.pdf",
            options=["A. 加热", "B. 规整", "C. 分拣", "D. 称重"],
            answer="B",
        )

    def get_question(self, stage, split, question_id):
        del stage, split, question_id
        return self.question

    def resolve_document(self, stage, split, document):
        del stage, split, document
        return Path("sample.pdf")


class _DocumentsStub:
    def __init__(self, page_count: int = 5) -> None:
        self.page_count = page_count
        self.rendered: list[int] = []
        self.render_scales: list[float] = []

    def render_page(self, path, page, scale=1.6):
        del path
        self.rendered.append(page)
        self.render_scales.append(scale)
        return f"page-{page}".encode()

    def metadata(self, path):
        del path
        return {"page_count": self.page_count}


def _settings(with_ocr: bool) -> SettingsStore:
    store = SettingsStore()
    payload = {
        "demo_mode": False,
        "vlm": {"base_url": "https://vlm.example/v1", "api_key": "secret", "model": "vlm-test"},
    }
    if with_ocr:
        payload["ocr"] = {"base_url": "https://ocr.example/v1", "api_key": "secret", "model": "ocr-test"}
    store.update(payload)
    return store


def test_preliminary_vision_baseline_skips_ocr_and_only_calls_vlm(monkeypatch) -> None:
    calls = []

    async def fake_chat(self, prompt, system_prompt, image_png=None, max_tokens=1600):
        del system_prompt, max_tokens
        calls.append({"service": self.service, "prompt": prompt, "image": image_png})
        return {
            "content": json.dumps({"answer": "B", "rationale": "目标图显示规整机构。", "analysis_summary": ["仅检查目标页设计图。"]}, ensure_ascii=False),
            "model": "vlm-test",
            "usage": {"total_tokens": 20},
            "duration_ms": 15.0,
        }

    monkeypatch.setattr(analysis_module.OpenAICompatibleClient, "chat", fake_chat)
    documents = _DocumentsStub()
    pipeline = AnalysisPipeline(_RepositoryStub(), documents, _settings(with_ocr=False), RunStore())
    response = asyncio.run(
        pipeline.analyze(
            AnalyzeRequest(stage="preliminary", split="train", question_id=7, page=3, force_demo=False, use_ocr=False)
        )
    )

    result = response["result"]
    assert [call["service"] for call in calls] == ["vlm"]
    assert documents.rendered == [3]
    assert documents.render_scales == [3.0]
    assert result["ocr_enabled"] is False
    assert result["ocr_pages"] == []
    assert result["ocr_text"] == ""
    assert result["metrics"]["accuracy"] == 1.0
    assert next(item for item in result["technical_trace"] if item["name"] == "ocr")["status"] == "skipped"
    assert "未启用" in calls[0]["prompt"]


def test_vlm_empty_final_content_retries_once_with_larger_budget(monkeypatch) -> None:
    calls = []

    async def fake_chat(self, prompt, system_prompt, image_png=None, max_tokens=1600, json_mode=False):
        del system_prompt
        calls.append({"prompt": prompt, "max_tokens": max_tokens, "json_mode": json_mode, "image": image_png})
        if len(calls) == 1:
            raise AppError("MODEL_EMPTY_RESPONSE", "模型返回了空内容", 502, True)
        return {
            "content": json.dumps(
                {
                    "answer": "B",
                    "answer_explanation": "重试后返回完整答案。",
                    "evidence": [{"page": 3, "source": "target_image", "quote": "规整结构", "relevance": "支持 B"}],
                    "option_analysis": [
                        {"option": option, "verdict": "selected" if option == "B" else "rejected", "reason": "重试结果"}
                        for option in "ABCD"
                    ],
                    "analysis_summary": ["结构化重试成功。"],
                },
                ensure_ascii=False,
            ),
            "model": "vlm-test",
            "usage": None,
            "duration_ms": 20.0,
        }

    monkeypatch.setattr(analysis_module.OpenAICompatibleClient, "chat", fake_chat)
    pipeline = AnalysisPipeline(_RepositoryStub(), _DocumentsStub(), _settings(with_ocr=False), RunStore())
    response = asyncio.run(
        pipeline.analyze(
            AnalyzeRequest(stage="preliminary", split="train", question_id=7, page=3, force_demo=False, use_ocr=False)
        )
    )

    assert len(calls) == 2
    assert calls[0]["max_tokens"] == 2400
    assert calls[1]["max_tokens"] == 3600
    assert calls[0]["json_mode"] is True
    assert calls[1]["json_mode"] is False
    assert "上一次响应没有形成可用的最终 JSON" in calls[1]["prompt"]
    assert response["result"]["answer"] == "B"
    assert response["result"]["vlm_retry_count"] == 1


def test_preliminary_ocr_context_uses_other_pages_and_target_image(monkeypatch) -> None:
    calls = []

    async def fake_chat(self, prompt, system_prompt, image_png=None, max_tokens=1600):
        del system_prompt, max_tokens
        calls.append({"service": self.service, "prompt": prompt, "image": image_png})
        if self.service == "ocr":
            return {"content": f"识别内容 {image_png.decode()}", "model": "ocr-test", "usage": None, "duration_ms": 10.0}
        return {
            "content": json.dumps({"answer": "B", "rationale": "其他页文字说明该机构用于规整。", "analysis_summary": ["读取其他页面功能描述。", "结合第3页结构图匹配 B。"]}, ensure_ascii=False),
            "model": "vlm-test",
            "usage": {"total_tokens": 60},
            "duration_ms": 20.0,
        }

    monkeypatch.setattr(analysis_module.OpenAICompatibleClient, "chat", fake_chat)
    documents = _DocumentsStub(page_count=5)
    pipeline = AnalysisPipeline(_RepositoryStub(), documents, _settings(with_ocr=True), RunStore())
    saved_cache = {}
    monkeypatch.setattr(pipeline.rag.index, "load_ocr_cache", lambda path, model: {})
    monkeypatch.setattr(
        pipeline.rag.index,
        "load_txt_ocr_corpus",
        lambda path, model: {"pages": {}, "meta": {"source": "none", "stale": False, "stale_reason": None}},
    )
    monkeypatch.setattr(
        pipeline.rag.index,
        "save_ocr_cache",
        lambda path, model, pages: saved_cache.update(pages),
    )
    response = asyncio.run(
        pipeline.analyze(
            AnalyzeRequest(stage="preliminary", split="train", question_id=7, page=3, force_demo=False, use_ocr=True)
        )
    )

    result = response["result"]
    assert [call["service"] for call in calls] == ["ocr", "ocr", "ocr", "ocr", "vlm"]
    assert documents.rendered == [3, 1, 2, 4, 5]
    assert documents.render_scales == [3.0, 1.6, 1.6, 1.6, 1.6]
    assert result["ocr_enabled"] is True
    assert result["ocr_pages"] == [1, 2, 4, 5]
    assert "===== PDF P.1 =====" in result["ocr_text"]
    assert "===== PDF P.5 =====" in calls[-1]["prompt"]
    assert calls[-1]["image"] == b"page-3"
    assert result["usage"]["ocr"]["calls"] == 4
    assert result["execution"]["context_strategy"] == "目标页 P.3 图像 + 其他页 OCR"
    assert sorted(saved_cache) == [1, 2, 4, 5]


def test_context_page_selection_is_nearest_first_but_returned_in_pdf_order() -> None:
    assert AnalysisPipeline._select_context_pages(12, 6, limit=4) == [4, 5, 7, 8]
    assert AnalysisPipeline._select_context_pages(3, 2) == [1, 3]
    assert AnalysisPipeline._select_context_pages(1, 1) == []


def test_semi_final_real_analysis_injects_rag_context_and_reports_models(monkeypatch) -> None:
    calls = []
    system_prompts = []

    class SemiFinalRepository(_RepositoryStub):
        def __init__(self) -> None:
            self.question = QuestionRecord(
                stage="semi_final",
                split="train",
                id=9,
                question="该装置如何实现定位？",
                document="sample.pdf",
                answer="通过定位挡板实现",
            )

    class RagStub:
        async def search(self, question, path, retrieval_config, reranker_config, ocr_config, **kwargs):
            del path, ocr_config
            assert question == "该装置如何实现定位？"
            assert retrieval_config.model == "embedding-test"
            assert reranker_config.model == "reranker-test"
            assert kwargs["target_page"] == 3
            assert "目标页识别" in kwargs["target_ocr"]
            return {
                "items": [
                    {
                        "id": "p2-c1",
                        "page": 2,
                        "source": "pdf_text",
                        "text": "定位挡板与驱动件配合实现定位。",
                        "retrieval_score": 0.8,
                        "rerank_score": 0.95,
                    }
                ],
                "retrieval_protocol": "openai_embeddings",
                "rerank_protocol": "openai_chat_completions",
                "chunk_count": 12,
                "recalled_count": 6,
                "reranked_count": 1,
                "corpus_sources": ["pdf_text", "ocr"],
                "pdf_text_pages": [1, 2],
                "ocr_pages": [3],
            }

    async def fake_chat(self, prompt, system_prompt, image_png=None, max_tokens=1600, json_mode=False):
        del image_png, max_tokens, json_mode
        calls.append((self.service, prompt))
        if self.service == "ocr":
            return {"content": "目标页识别：定位挡板", "model": "ocr-test", "usage": None, "duration_ms": 5.0}
        assert "复赛 RAG 检索增强上下文" in prompt
        assert "定位挡板与驱动件配合实现定位" in prompt
        system_prompts.append(system_prompt)
        return {
            "content": json.dumps(
                {
                    "answer": "通过定位挡板实现",
                    "rationale": "目标页与检索片段相互印证。",
                    "evidence": [{"page": 2, "source": "ocr", "quote": "定位挡板", "relevance": "支持答案"}],
                },
                ensure_ascii=False,
            ),
            "model": "vlm-test",
            "usage": {"total_tokens": 30},
            "duration_ms": 10.0,
        }

    monkeypatch.setattr(analysis_module.OpenAICompatibleClient, "chat", fake_chat)
    settings = _settings(with_ocr=True)
    settings.update(
        {
            "retrieval": {"base_url": "https://retrieval.example/v1", "api_key": "secret", "model": "embedding-test"},
            "reranker": {"base_url": "https://reranker.example/v1", "api_key": "secret", "model": "reranker-test"},
        }
    )
    pipeline = AnalysisPipeline(SemiFinalRepository(), _DocumentsStub(), settings, RunStore(), rag=RagStub())
    response = asyncio.run(
        pipeline.analyze(
            AnalyzeRequest(stage="semi_final", split="train", question_id=9, page=3, force_demo=False, use_rag=True)
        )
    )

    result = response["result"]
    assert [service for service, _ in calls] == ["ocr", "vlm"]
    vlm_prompt = calls[-1][1]
    assert "这是复赛开放式问答，不存在候选选项" in vlm_prompt
    assert "选项：" not in vlm_prompt
    assert "不要生成 option_analysis" in vlm_prompt
    assert "answer 禁止包含" in vlm_prompt and "目标图页" in vlm_prompt
    assert "复赛开放式问答" in system_prompts[0]
    assert "禁止写选项字母、目标图页、PDF 页码" in system_prompts[0]
    assert result["rag"]["enabled"] is True
    assert result["rag"]["corpus_sources"] == ["pdf_text", "ocr"]
    assert result["models"]["retrieval"] == "embedding-test"
    assert result["models"]["reranker"] == "reranker-test"
    assert result["usage"]["retrieval"] == {"protocol": "openai_embeddings", "chunks": 12, "recalled": 6}
    assert result["usage"]["reranker"] == {"protocol": "openai_chat_completions", "reranked": 1}
    assert result["execution"]["context_strategy"] == "目标页 P.3 图像 + 目标页 OCR + 1 个 RAG 重排片段"
    trace = {item["name"]: item for item in result["technical_trace"]}
    assert "分块 12" in trace["indexing"]["output_summary"]
    assert "召回 6" in trace["retrieval"]["output_summary"]
    assert trace["retrieval"]["model"] == "embedding-test"
    assert "重排 1" in trace["reranking"]["output_summary"]
    assert trace["reranking"]["model"] == "reranker-test"


def test_semantic_evaluator_calculates_f1_and_proxy_score(monkeypatch) -> None:
    settings = _settings(with_ocr=False)
    settings.update({"vlm": {"model": "MiniMax-M3"}})
    pipeline = AnalysisPipeline(_RepositoryStub(), _DocumentsStub(), settings, RunStore())

    async def fake_chat(self, prompt, system_prompt, image_png=None, max_tokens=1600, json_mode=False):
        assert self.service == "semantic_evaluator"
        assert '"prediction"' in prompt and '"reference"' in prompt
        assert "Precision" in system_prompt
        assert image_png is None
        assert json_mode is True
        return {
            "content": json.dumps({"precision": 0.8, "recall": 0.5, "reason": "核心实体部分覆盖"}, ensure_ascii=False),
            "model": "MiniMax-M3",
            "usage": {"total_tokens": 24},
            "duration_ms": 6.0,
        }

    monkeypatch.setattr(analysis_module.OpenAICompatibleClient, "chat", fake_chat)
    result = {
        "answer": "预测答案",
        "usage": {},
        "metrics": {"applicable": True, "rouge1_f1": 0.6, "exact_match": 0.0},
    }
    asyncio.run(pipeline._add_semantic_metrics(result, "参考答案"))

    semantic_f1 = 2 * 0.8 * 0.5 / (0.8 + 0.5)
    assert result["metrics"]["semantic_f1"] == pytest.approx(semantic_f1)
    assert result["metrics"]["semantic_proxy_score"] == pytest.approx(0.5 * semantic_f1 + 0.25 * 0.6)
    # The LLM judge is never reported as BERTScore and never feeds the
    # official weighted score.
    assert result["metrics"]["bert_f1"] is None
    assert result["metrics"]["final_score"] is None
    assert result["metrics"]["bert_f1_status"] == "unavailable"
    assert result["metrics"]["semantic_status"] == "completed"
    assert result["usage"]["semantic_evaluator"] == {"total_tokens": 24}
