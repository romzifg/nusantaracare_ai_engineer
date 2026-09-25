from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=3, max_length=1500)

    @field_validator("question")
    @classmethod
    def clean_question(cls, value):
        value = value.strip()
        if len(value) < 3:
            raise ValueError("Pertanyaan terlalu pendek.")
        return value


class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    doc_title: str
    doc_version: str
    effective_date: str
    is_active: bool
    section_title: str
    line_start: int
    line_end: int
    quote: str


class Answer(BaseModel):
    answer: str
    confidence_label: Literal["high", "medium", "low"]
    reason_code: str
    citations: list[Citation] = Field(default_factory=list)
    route: str = "none"
    mode: str = "none"
    trace: list[str] = Field(default_factory=list)


class Selection(BaseModel):
    """Model memilih bukti; backend menyusun jawaban dari sumbernya."""
    model_config = ConfigDict(extra="forbid")
    selected_ids: list[str] = Field(max_length=3)
    abstain: bool
