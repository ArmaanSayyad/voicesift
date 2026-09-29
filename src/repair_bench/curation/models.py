from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

Label = Literal["interruption", "backchannel", "other_overlap", "uncertain"]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Turn(Strict):
    role: Literal["user", "assistant"]
    text: str = Field(min_length=1, max_length=12000)
    start_s: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    end_s: float | None = Field(default=None, ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def span(self):
        if not self.text.strip():
            raise ValueError("Empty turn")
        if (self.start_s is None) != (self.end_s is None):
            raise ValueError("Both timestamps are required")
        if self.start_s is not None and self.end_s <= self.start_s:
            raise ValueError("Invalid time span")
        return self


class Conversation(Strict):
    id: str = Field(min_length=1, max_length=120)
    source_group: str = Field(min_length=1, max_length=160)
    split: Literal["train", "dev", "test"] = "dev"
    provenance: str = Field(min_length=1, max_length=1000)
    timing_source: Literal[
        "speech_intervals", "transcript_segments", "manual_annotations"
    ] = "transcript_segments"
    turns: list[Turn] = Field(min_length=2, max_length=200)


class Review(Strict):
    label: Label
    agent_outcome: Literal["stopped", "continued", "resumed", "unknown"] = "unknown"
    include: bool = False
    note: str = Field(default="", max_length=2000)
    reviewer: str = Field(min_length=1, max_length=80)
    expected_version: int = Field(ge=0)

    @model_validator(mode="after")
    def inclusion(self):
        if self.include and self.label != "interruption":
            raise ValueError("Only reviewer-confirmed interruptions can be included")
        return self


class Import(Strict):
    jsonl: str = Field(min_length=1, max_length=2_000_000)


class Analyze(Strict):
    candidate_ids: list[str] = Field(min_length=1, max_length=50)
