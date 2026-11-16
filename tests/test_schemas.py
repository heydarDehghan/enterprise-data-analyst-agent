import pytest


def test_chat_request_serializes_required_and_default_optional_fields() -> None:
    """A new chat requires a question and has no conversation ID yet."""
    from app.schemas import ChatRequest

    request = ChatRequest(question="Compare September and October revenue.")

    assert request.model_dump() == {
        "question": "Compare September and October revenue.",
        "conversation_id": None,
    }


def test_chat_request_accepts_an_optional_conversation_id() -> None:
    """An existing conversation can be continued explicitly."""
    from app.schemas import ChatRequest

    request = ChatRequest(
        question="Now compare the customer segments.",
        conversation_id="conv-001",
    )

    assert request.model_dump() == {
        "question": "Now compare the customer segments.",
        "conversation_id": "conv-001",
    }


def test_chat_request_rejects_missing_or_blank_questions() -> None:
    """The question is required and whitespace-only input is invalid."""
    from app.schemas import ChatRequest

    with pytest.raises(ValueError):
        ChatRequest()

    with pytest.raises(ValueError):
        ChatRequest(question="   ")


def test_artifact_metadata_validates_fields_and_serializes_optionals() -> None:
    """Artifacts carry stable metadata without requiring a download URL."""
    from app.schemas import ArtifactMetadata

    artifact = ArtifactMetadata(
        artifact_id="artifact-001",
        kind="chart",
        title="October paid revenue",
        media_type="image/png",
    )

    assert artifact.model_dump() == {
        "artifact_id": "artifact-001",
        "kind": "chart",
        "title": "October paid revenue",
        "media_type": "image/png",
        "url": None,
    }

    with_url = ArtifactMetadata(
        artifact_id="artifact-002",
        kind="table",
        title="Monthly paid revenue",
        media_type="text/csv",
        url="/artifacts/artifact-002",
    )

    assert with_url.model_dump()["url"] == "/artifacts/artifact-002"

    with pytest.raises(ValueError):
        ArtifactMetadata(
            kind="chart",
            title="Missing identifier",
            media_type="image/png",
        )


def test_tool_trace_validates_fields_and_serializes_optionals() -> None:
    """A trace identifies the ordered tool step and its outcome."""
    from app.schemas import ToolTrace

    trace = ToolTrace(step=1, tool_name="query_sqlite", status="success")

    assert trace.model_dump() == {
        "step": 1,
        "tool_name": "query_sqlite",
        "status": "success",
        "duration_ms": None,
        "error": None,
    }

    failed_trace = ToolTrace(
        step=2,
        tool_name="python_sandbox",
        status="error",
        duration_ms=125,
        error="execution failed",
    )

    assert failed_trace.model_dump()["duration_ms"] == 125
    assert failed_trace.model_dump()["error"] == "execution failed"

    with pytest.raises(ValueError):
        ToolTrace(step=1, status="success")


def test_chat_response_requires_conversation_answer_and_nonblank_trace_id() -> None:
    """Every response has conversation identity, trace identity, and an answer."""
    from app.schemas import ChatResponse

    with pytest.raises(ValueError):
        ChatResponse(
            trace_id="trace-001",
            answer="Missing conversation identity.",
        )

    with pytest.raises(ValueError):
        ChatResponse(conversation_id="conv-001", trace_id="trace-001")

    with pytest.raises(ValueError):
        ChatResponse(
            conversation_id="conv-001",
            answer="Missing trace identity.",
        )

    with pytest.raises(ValueError):
        ChatResponse(
            conversation_id="conv-001",
            trace_id="   ",
            answer="Blank trace identity.",
        )


def test_chat_response_serializes_default_empty_collections() -> None:
    """Artifacts and tool traces are optional and default to empty lists."""
    from app.schemas import ChatResponse

    response = ChatResponse(
        conversation_id="conv-001",
        trace_id="trace-001",
        answer="October paid revenue increased.",
    )

    assert response.model_dump() == {
        "conversation_id": "conv-001",
        "trace_id": "trace-001",
        "answer": "October paid revenue increased.",
        "artifacts": [],
        "trace": [],
    }


def test_chat_response_serializes_nested_artifacts_and_tool_traces() -> None:
    """Nested response metadata serializes into JSON-compatible dictionaries."""
    from app.schemas import ArtifactMetadata, ChatResponse, ToolTrace

    response = ChatResponse(
        conversation_id="conv-002",
        trace_id="trace-002",
        answer="A chart was generated from paid orders only.",
        artifacts=[
            ArtifactMetadata(
                artifact_id="artifact-001",
                kind="chart",
                title="Paid revenue comparison",
                media_type="image/png",
                url="/artifacts/artifact-001",
            )
        ],
        trace=[
            ToolTrace(
                step=1,
                tool_name="query_sqlite",
                status="success",
                duration_ms=25,
            )
        ],
    )

    assert response.model_dump() == {
        "conversation_id": "conv-002",
        "trace_id": "trace-002",
        "answer": "A chart was generated from paid orders only.",
        "artifacts": [
            {
                "artifact_id": "artifact-001",
                "kind": "chart",
                "title": "Paid revenue comparison",
                "media_type": "image/png",
                "url": "/artifacts/artifact-001",
            }
        ],
        "trace": [
            {
                "step": 1,
                "tool_name": "query_sqlite",
                "status": "success",
                "duration_ms": 25,
                "error": None,
            }
        ],
    }
