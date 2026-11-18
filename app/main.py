"""Authenticated HTTP boundary for tenant-scoped analysis conversations."""

import json
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from starlette.datastructures import State

from app.agent import (
    AgentLoop, AgentOperationFailed, AgentStepLimitExceeded,
    EnvChatCompletionsClient, ProviderBudgetExceeded, UngroundedAnswerError,
)
from app.config import LLMConfigurationError, load_llm_budget, load_llm_config
from app.sandbox import DockerPythonSandbox
from app.schemas import ChatRequest, ChatResponse
from app.secure_data import SecureDataAccess, TenantContext, get_tenant_context
from app.state import (
    ArtifactNotFoundError, ConversationNotFoundError, SQLiteConversationStore,
)


class _ServiceState(State):
    """Keep injectable service slots visible to attribute restoration tools."""

    @property
    def llm_client(self) -> Any:
        return self._state.get("llm_client")

    @llm_client.setter
    def llm_client(self, value: Any) -> None:
        self._state["llm_client"] = value

    @property
    def conversation_store(self) -> Any:
        return self._state.get("conversation_store")

    @conversation_store.setter
    def conversation_store(self, value: Any) -> None:
        self._state["conversation_store"] = value


def health() -> dict[str, str]:
    return {"status": "ok"}


def chat(
    body: ChatRequest,
    request: Request,
    tenant: TenantContext = Depends(get_tenant_context),
) -> ChatResponse:
    state = request.app.state
    store = getattr(state, "conversation_store", None)
    llm = getattr(state, "llm_client", None)
    schema_text = getattr(state, "schema_text", None)
    if (not isinstance(store, SQLiteConversationStore)
            or not callable(getattr(llm, "create", None))
            or not isinstance(schema_text, str)):
        raise HTTPException(status_code=503, detail="Chat service unavailable")

    try:
        calls, tokens = load_llm_budget()
    except LLMConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    loop = AgentLoop(
        llm=llm, tenant=tenant, schema_text=schema_text,
        sandbox=getattr(state, "sandbox", None), state_store=store,
        max_llm_calls=calls, max_total_tokens=tokens,
        purchase_drops_enabled=getattr(state, "purchase_drops_enabled", False),
    )
    try:
        return loop.run(body)
    except ConversationNotFoundError:
        raise HTTPException(status_code=404, detail="Conversation not found") from None
    except (ProviderBudgetExceeded, AgentStepLimitExceeded):
        raise HTTPException(status_code=429, detail="Request budget exceeded") from None
    except AgentOperationFailed:
        raise HTTPException(status_code=503, detail="Analysis provider unavailable") from None
    except UngroundedAnswerError:
        raise HTTPException(status_code=502, detail="Analysis answer rejected") from None


def get_artifact(
    artifact_id: str,
    request: Request,
    tenant: TenantContext = Depends(get_tenant_context),
) -> Response:
    store = getattr(request.app.state, "conversation_store", None)
    if not isinstance(store, SQLiteConversationStore):
        raise HTTPException(status_code=503, detail="Artifact storage unavailable")
    try:
        content = store.get_artifact(tenant, artifact_id)
    except ArtifactNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Artifact not found") from exc
    return Response(content=content, media_type="image/png")


def _register_routes(service: FastAPI) -> None:
    service.add_api_route("/health", health, methods=["GET"])
    service.add_api_route("/chat", chat, methods=["POST"], response_model=ChatResponse)
    service.add_api_route("/artifacts/{artifact_id}", get_artifact, methods=["GET"])


def create_app(
    *,
    data_dir: Path,
    token_map_path: Path,
    schema_path: Path,
    state_db_path: Path,
    artifact_root: Path,
    docker_image: str,
    llm_client: Any | None = None,
) -> FastAPI:
    """Build an independently configured service without global test state."""
    token_map = json.loads(Path(token_map_path).read_text(encoding="utf-8"))
    if not isinstance(token_map, dict) or not token_map:
        raise ValueError("Invalid tenant token mapping")
    schema_text = Path(schema_path).read_text(encoding="utf-8")
    if not schema_text.strip():
        raise ValueError("Schema reference is empty")

    load_llm_budget()
    if llm_client is None:
        load_llm_config()
        llm_client = EnvChatCompletionsClient()

    service = FastAPI(title="Kamidso Lite")
    service.state = _ServiceState()
    service.state.secure_data = SecureDataAccess(data_dir=data_dir, token_map=token_map)
    service.state.conversation_store = SQLiteConversationStore(
        db_path=state_db_path, artifact_root=artifact_root,
    )
    service.state.schema_text = schema_text
    service.state.sandbox = DockerPythonSandbox(image=docker_image, artifact_root=artifact_root)
    service.state.llm_client = llm_client
    service.state.purchase_drops_enabled = True
    _register_routes(service)
    return service


# Existing callers can inject collaborators on this app.state. Deployments
# should construct their configured application with create_app().
app = FastAPI(title="Kamidso Lite")
app.state = _ServiceState()
_register_routes(app)
app.state.secure_data = None
app.state.conversation_store = None
app.state.schema_text = None
app.state.sandbox = None
app.state.llm_client = None
