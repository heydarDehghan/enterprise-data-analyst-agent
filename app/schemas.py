from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator


NonBlankText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChatRequest(ApiModel):
    question: NonBlankText
    conversation_id: NonBlankText | None = None


class ArtifactMetadata(ApiModel):
    artifact_id: NonBlankText
    kind: NonBlankText
    title: NonBlankText
    media_type: NonBlankText
    url: NonBlankText | None = None


class ToolTrace(ApiModel):
    step: int = Field(ge=1)
    tool_name: NonBlankText
    status: NonBlankText
    duration_ms: int | None = Field(default=None, ge=0)
    error: str | None = None


class ChatResponse(ApiModel):
    conversation_id: NonBlankText
    trace_id: NonBlankText
    answer: str = Field(min_length=1)
    artifacts: list[ArtifactMetadata] = Field(default_factory=list)
    trace: list[ToolTrace] = Field(default_factory=list)

    @field_validator("answer")
    @classmethod
    def validate_answer(cls, answer: str) -> str:
        if not answer.strip():
            raise ValueError("Answer must contain text")
        return answer
