import pytest

from app.services.evaluation import (
    evaluate_answer,
    exact_match,
    final_score,
    normalize_answer,
    normalize_choice,
    rouge1_f1,
)


def test_answer_normalization_handles_width_case_whitespace_and_punctuation() -> None:
    assert normalize_answer("  Ａ，部件。\n") == "a部件"
    assert normalize_answer(None) == ""
    assert exact_match("第一锥齿轮。", " 第一锥齿轮 ") == 0.0


def test_open_answer_exact_match_is_strict_after_nfkc_and_outer_trim() -> None:
    prediction = "酯化反应的主要原料是：丁二酸和 1,4-丁二醇。"
    assert exact_match(prediction, "丁二酸和1,4-丁二醇") == 0.0
    assert exact_match(" 磁性板(24) ", "磁性板（24）") == 1.0


def test_open_answer_exact_match_rejects_incomplete_multi_entity_answer() -> None:
    assert exact_match("丁二酸", "丁二酸和1,4-丁二醇") == 0.0


@pytest.mark.parametrize(
    ("prediction", "reference"),
    [
        ("丁二酸", "丁二酸和1,4-丁二醇"),
        ("部件7、部件8、部件9", "部件7，部件8"),
        ("第一锥齿轮及其他部件", "第一锥齿轮"),
    ],
)
def test_semifinal_exact_match_rejects_incomplete_or_extra_answers(
    prediction: str, reference: str
) -> None:
    assert exact_match(prediction, reference) == 0.0


def test_final_score_uses_competition_weights() -> None:
    assert final_score(0.8, 0.6, 1.0) == pytest.approx(0.8)
    assert final_score(None, 0.6, 1.0) is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("B", "B"),
        ("答案为 B。", "B"),
        ("选择：d", "D"),
        ("无法判断", ""),
        (None, ""),
    ],
)
def test_choice_normalization(raw: str | None, expected: str) -> None:
    assert normalize_choice(raw) == expected


def test_metrics_are_applicable_only_when_reference_exists() -> None:
    missing = evaluate_answer("semi_final", "任意答案", None)
    choice = evaluate_answer("preliminary", "答案是 C", "C")

    assert missing == {
        "applicable": False,
        "accuracy": None,
        "exact_match": None,
        "rouge1_f1": None,
        "semantic_f1": None,
        "bert_f1": None,
        "final_score": None,
    }
    assert choice["accuracy"] == 1.0
    assert choice["exact_match"] is None


def test_rouge1_f1_rewards_token_overlap() -> None:
    exact = rouge1_f1("用于快速组装定位", "用于快速组装定位")
    partial = rouge1_f1("用于快速定位", "用于快速组装定位")
    none = rouge1_f1("完全无关", "用于快速组装定位")

    assert exact == pytest.approx(1.0)
    assert 0.0 < partial < 1.0
    assert none == 0.0
