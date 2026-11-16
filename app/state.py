"""SQLite-backed, tenant-bound conversation and artifact state."""

import json
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from app.evidence_history import MAX_EVIDENCE_TURNS, encode_turn_evidence, recover_turn_evidence
from app.schemas import ArtifactMetadata, ToolTrace
from app.secure_data import TenantContext
from app.structured_grounding import EvidenceScope, RecordedToolResult


_OPAQUE_ID = re.compile(r"[0-9a-f]{32}\Z")
_MAX_ARTIFACT_BYTES = 10 * 1024 * 1024


class ConversationNotFoundError(LookupError):
    """An unknown, invalid, or foreign conversation has the same outcome."""


class ArtifactNotFoundError(LookupError):
    """An artifact is unavailable to the authenticated tenant."""


@dataclass(frozen=True)
class ConversationTurn:
    trace_id: str
    status: str
    answer: str | None
    trace: list[ToolTrace]
    failure_code: str | None = None


@dataclass(frozen=True)
class ConversationSnapshot:
    messages: list[dict]
    turns: list[ConversationTurn]
    artifacts: list[ArtifactMetadata]
    evidence: list[RecordedToolResult]


def _tenant_key(tenant: TenantContext) -> str:
    if not isinstance(tenant, TenantContext):
        raise TypeError("Authenticated tenant context required")
    return tenant.state_key


class SQLiteConversationStore:
    def __init__(self, *, db_path: Path, artifact_root: Path) -> None:
        self.db_path = Path(db_path).resolve()
        self.artifact_root = Path(artifact_root).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_root.mkdir(parents=True, exist_ok=True)

        # Never turn a supplied source SQLite database into a state database.
        if self.db_path.exists() and self.db_path.stat().st_size:
            uri = f"file:{self.db_path.as_posix()}?mode=ro&immutable=1"
            with closing(sqlite3.connect(uri, uri=True)) as existing:
                row = existing.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'conversations'"
                ).fetchone()
                if row is None:
                    raise ValueError("Existing database is not a conversation store")

        with closing(self._connect()) as connection, connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    tenant_key TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS turns (
                    turn_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    trace_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    answer TEXT,
                    failure_code TEXT,
                    trace_json TEXT NOT NULL,
                    messages_json TEXT NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
                );
                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    tenant_key TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
                );
                CREATE INDEX IF NOT EXISTS idx_turns_conversation
                    ON turns(conversation_id, turn_id);
                CREATE INDEX IF NOT EXISTS idx_artifacts_conversation
                    ON artifacts(conversation_id);
                CREATE TABLE IF NOT EXISTS turn_evidence (
                    turn_id INTEGER PRIMARY KEY,
                    proof_json TEXT NOT NULL,
                    FOREIGN KEY (turn_id) REFERENCES turns(turn_id) ON DELETE CASCADE
                );
            """)
            columns = {row[1] for row in connection.execute("PRAGMA table_info(turns)")}
            if "failure_code" not in columns:
                connection.execute("ALTER TABLE turns ADD COLUMN failure_code TEXT")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=5)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _require_id(conversation_id: str) -> None:
        if not isinstance(conversation_id, str) or not _OPAQUE_ID.fullmatch(conversation_id):
            raise ConversationNotFoundError("Conversation not found")

    def create_conversation(self, tenant: TenantContext) -> str:
        key = _tenant_key(tenant)
        conversation_id = uuid4().hex
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "INSERT INTO conversations (conversation_id, tenant_key) VALUES (?, ?)",
                (conversation_id, key),
            )
        return conversation_id

    def get_conversation(
        self, tenant: TenantContext, conversation_id: str
    ) -> ConversationSnapshot:
        key = _tenant_key(tenant)
        self._require_id(conversation_id)
        with closing(self._connect()) as connection:
            owner = connection.execute(
                "SELECT 1 FROM conversations WHERE conversation_id = ? AND tenant_key = ?",
                (conversation_id, key),
            ).fetchone()
            if owner is None:
                raise ConversationNotFoundError("Conversation not found")
            stored_turns = connection.execute(
                "SELECT t.trace_id, t.status, t.answer, t.failure_code, t.trace_json, t.messages_json, "
                "e.proof_json FROM turns AS t LEFT JOIN turn_evidence AS e USING (turn_id) "
                "WHERE t.conversation_id = ? ORDER BY t.turn_id",
                (conversation_id,),
            ).fetchall()
            stored_artifacts = connection.execute(
                "SELECT metadata_json FROM artifacts "
                "WHERE conversation_id = ? AND tenant_key = ? ORDER BY rowid",
                (conversation_id, key),
            ).fetchall()

        messages: list[dict] = []
        turns: list[ConversationTurn] = []
        evidence: list[RecordedToolResult] = []
        scope = EvidenceScope(key, conversation_id)
        for row in stored_turns:
            turn_messages = json.loads(row["messages_json"])
            trace = [ToolTrace.model_validate(item) for item in json.loads(row["trace_json"])]
            messages.extend(turn_messages)
            turns.append(ConversationTurn(
                trace_id=row["trace_id"], status=row["status"], answer=row["answer"],
                trace=trace, failure_code=row["failure_code"],
            ))
            if row["status"] == "success":
                evidence.extend(recover_turn_evidence(scope, row["proof_json"], turn_messages, trace))
        return ConversationSnapshot(
            messages=messages,
            turns=turns,
            artifacts=[ArtifactMetadata.model_validate(json.loads(row["metadata_json"]))
                       for row in stored_artifacts],
            evidence=evidence,
        )

    def save_turn(
        self,
        tenant: TenantContext,
        conversation_id: str,
        *,
        trace_id: str,
        status: str,
        answer: str | None,
        trace: list[ToolTrace],
        messages: list[dict],
        artifacts: list[ArtifactMetadata],
        failure_code: str | None = None,
        evidence: list[RecordedToolResult] | None = None,
    ) -> None:
        key = _tenant_key(tenant)
        self._require_id(conversation_id)
        if status not in ("success", "failed"):
            raise ValueError("Invalid turn status")
        if failure_code not in (None, "non_persian", "unsupported_number", "invalid_date_range", "invalid_template"):
            raise ValueError("Invalid failure category")
        if status == "success" and failure_code is not None:
            raise ValueError("Successful turn cannot have a failure category")
        proof = (
            encode_turn_evidence(EvidenceScope(key, conversation_id), evidence, messages, trace)
            if status == "success" and evidence else None
        )
        for artifact in artifacts:
            self._artifact_bytes(artifact.artifact_id)

        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            owner = connection.execute(
                "SELECT 1 FROM conversations WHERE conversation_id = ? AND tenant_key = ?",
                (conversation_id, key),
            ).fetchone()
            if owner is None:
                raise ConversationNotFoundError("Conversation not found")
            turn = connection.execute(
                "INSERT INTO turns (conversation_id, trace_id, status, answer, failure_code, "
                "trace_json, messages_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (conversation_id, trace_id, status, answer, failure_code,
                 json.dumps([item.model_dump() for item in trace], ensure_ascii=False),
                 json.dumps(messages, ensure_ascii=False)),
            )
            if proof is not None:
                connection.execute(
                    "INSERT INTO turn_evidence (turn_id, proof_json) VALUES (?, ?)",
                    (turn.lastrowid, proof),
                )
                connection.execute(
                    "DELETE FROM turn_evidence WHERE turn_id IN ("
                    "SELECT t.turn_id FROM turns AS t JOIN turn_evidence USING (turn_id) "
                    "WHERE t.conversation_id = ? ORDER BY t.turn_id DESC LIMIT -1 OFFSET ?)",
                    (conversation_id, MAX_EVIDENCE_TURNS),
                )
            for artifact in artifacts:
                connection.execute(
                    "INSERT INTO artifacts (artifact_id, conversation_id, tenant_key, "
                    "metadata_json) VALUES (?, ?, ?, ?)",
                    (artifact.artifact_id, conversation_id, key,
                     artifact.model_dump_json()),
                )

    def _artifact_bytes(self, artifact_id: str) -> bytes:
        if not isinstance(artifact_id, str) or not _OPAQUE_ID.fullmatch(artifact_id):
            raise ArtifactNotFoundError("Artifact not found")
        path = self.artifact_root / f"{artifact_id}.png"
        if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_ARTIFACT_BYTES:
            raise ArtifactNotFoundError("Artifact not found")
        # A stored file can change after registration; reject altered content.
        from app.sandbox import _valid_png
        content = path.read_bytes()
        if len(content) > _MAX_ARTIFACT_BYTES or not _valid_png(content):
            raise ArtifactNotFoundError("Artifact not found")
        return content

    def get_artifact(self, tenant: TenantContext, artifact_id: str) -> bytes:
        key = _tenant_key(tenant)
        if not isinstance(artifact_id, str) or not _OPAQUE_ID.fullmatch(artifact_id):
            raise ArtifactNotFoundError("Artifact not found")
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT metadata_json FROM artifacts "
                "WHERE artifact_id = ? AND tenant_key = ?",
                (artifact_id, key),
            ).fetchone()
        if row is None:
            raise ArtifactNotFoundError("Artifact not found")
        metadata = ArtifactMetadata.model_validate_json(row["metadata_json"])
        if metadata.media_type != "image/png":
            raise ArtifactNotFoundError("Artifact not found")
        return self._artifact_bytes(artifact_id)
