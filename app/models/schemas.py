from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


Stage = Literal["preliminary", "semi_final"]
Split = Literal["train", "test"]


class EndpointSettingsUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: str | None = None
    api_key: str | None = None
    clear_api_key: bool = False
    model: str | None = None
    timeout: float | None = Field(default=None, ge=1, le=600)
    temperature: float | None = Field(default=None, ge=0, le=2)


class ModelSettingsUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    demo_mode: bool | None = None
    ocr: EndpointSettingsUpdate | None = None
    vlm: EndpointSettingsUpdate | None = None
    retrieval: EndpointSettingsUpdate | None = None
    reranker: EndpointSettingsUpdate | None = None


ModelService = Literal["ocr", "vlm", "retrieval", "reranker", "both", "all"]


class ModelTestRequest(BaseModel):
    service: ModelService = "both"
    target: ModelService | None = None
    kind: ModelService | None = None

    @property
    def selected_service(self) -> ModelService:
        return self.target or self.kind or self.service


class AnalyzeRequest(BaseModel):
    stage: Stage
    split: Split
    question_id: str | int
    page: int | None = Field(default=None, ge=1)
    force_demo: bool | None = None
    use_ocr: bool = True
    use_rag: bool = True
    # Opt-in LLM semantic review. It is an extra model call and is not the
    # official BERTScore, so it stays off unless explicitly requested.
    evaluate_semantic: bool = False


class JobCreateRequest(BaseModel):
    stage: Stage
    split: Split
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=20, ge=1, le=500)
    concurrency: int = Field(default=1, ge=1, le=4)
    question_ids: list[str | int] | None = None
    force_demo: bool | None = None
    use_ocr: bool = True
    use_rag: bool = True
    evaluate_semantic: bool = False


class QuestionRecord(BaseModel):
    stage: Stage
    split: Split
    id: str | int
    question: str
    document: str
    options: list[str] | None = None
    answer: str | None = None
    group: int | str | None = None
    page_hint: int | None = None
    has_reference: bool = False
    document_available: bool = False
    raw: dict[str, Any] = Field(default_factory=dict, exclude=True)
