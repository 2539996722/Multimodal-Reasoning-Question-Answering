from __future__ import annotations

import re
import unicodedata
from typing import Any

from rouge_score import rouge_scorer

try:
    import jieba

    jieba.setLogLevel(20)
except ImportError:  # pragma: no cover - optional fallback
    jieba = None


def normalize_answer(value: str | None) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    return "".join(char for char in text if not char.isspace() and not unicodedata.category(char).startswith("P"))


ANSWER_PREFIX_PATTERN = re.compile(
    r"^\s*(?:(?:最终|标准|参考)\s*)?(?:答案|回答|答复|结论|结果|answer|final\s+answer)"
    r"\s*(?:(?:是|为|如下)\s*)?(?:[:：=\-]\s*)?",
    flags=re.IGNORECASE,
)
def strip_answer_prefix(value: str | None) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip()
    previous = None
    while text and text != previous:
        previous = text
        text = ANSWER_PREFIX_PATTERN.sub("", text, count=1).strip()
    return text


def normalize_choice(value: str | None) -> str:
    if not value:
        return ""
    match = re.search(r"(?<![A-Z])[ABCD](?![A-Z])", str(value).upper())
    return match.group(0) if match else ""


def _normalize_exact(value: str | None) -> str:
    if value is None:
        return ""
    return unicodedata.normalize("NFKC", str(value)).casefold().strip()


def exact_match(prediction: str | None, reference: str | None) -> float | None:
    if reference is None:
        return None
    return float(_normalize_exact(prediction) == _normalize_exact(reference))


class _JiebaTokenizer:
    def tokenize(self, text: str) -> list[str]:
        normalized = normalize_answer(text)
        if not normalized:
            return []
        if jieba is not None:
            return [token.strip() for token in jieba.cut(normalized) if token.strip()]
        return list(normalized)


_ROUGE_SCORER = rouge_scorer.RougeScorer(
    ["rouge1"],
    use_stemmer=False,
    tokenizer=_JiebaTokenizer(),
)


def rouge1_f1(prediction: str | None, reference: str | None) -> float | None:
    if reference is None:
        return None
    score = _ROUGE_SCORER.score(str(reference or ""), str(prediction or ""))["rouge1"]
    return float(score.fmeasure)


def final_score(bert_f1: float | None, rouge1: float | None, em: float | None) -> float | None:
    """Weighted semifinal score: 0.5 semantic + 0.25 lexical + 0.25 exact."""

    if bert_f1 is None or rouge1 is None or em is None:
        return None
    return 0.5 * float(bert_f1) + 0.25 * float(rouge1) + 0.25 * float(em)


def evaluate_answer(stage: str, prediction: str | None, reference: str | None) -> dict[str, Any]:
    if reference is None:
        return {
            "applicable": False,
            "accuracy": None,
            "exact_match": None,
            "rouge1_f1": None,
            "semantic_f1": None,
            "bert_f1": None,
            "final_score": None,
        }
    if stage == "preliminary":
        accuracy = float(normalize_choice(prediction) == normalize_choice(reference))
        return {
            "applicable": True,
            "accuracy": accuracy,
            "exact_match": None,
            "rouge1_f1": None,
            "semantic_f1": None,
            "bert_f1": None,
            "final_score": None,
        }
    return {
        "applicable": True,
        "accuracy": None,
        "exact_match": exact_match(prediction, reference),
        "rouge1_f1": rouge1_f1(prediction, reference),
        "semantic_f1": None,
        "bert_f1": None,
        "final_score": None,
    }
