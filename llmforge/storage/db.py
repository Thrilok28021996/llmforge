"""SQLite database with async access via aiosqlite."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

import aiosqlite

from llmforge.config import db_path

logger = logging.getLogger(__name__)

SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS models (
    id TEXT PRIMARY KEY,
    backend TEXT NOT NULL,
    name TEXT NOT NULL,
    quantization TEXT,
    parameter_count INTEGER,
    context_length INTEGER,
    size_bytes INTEGER,
    capabilities TEXT,
    last_used_at TEXT
);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    prompt TEXT NOT NULL,
    response TEXT,
    parameters TEXT NOT NULL,
    ttft_ms REAL,
    tokens_per_second REAL,
    total_latency_ms REAL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    hw_gpu_util_avg REAL,
    hw_cpu_util_avg REAL,
    hw_ram_used_gb REAL,
    hw_device TEXT,
    experiment_id TEXT,
    comparison_id TEXT,
    score_llm_judge REAL,
    score_bleu REAL,
    score_rouge_l REAL,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS prompt_templates (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    content TEXT NOT NULL,
    variables TEXT,
    version INTEGER DEFAULT 1,
    parent_id TEXT
);

CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    baseline_run_id TEXT
);

CREATE TABLE IF NOT EXISTS test_cases (
    id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    prompt TEXT NOT NULL,
    expected_output TEXT,
    rubric TEXT
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL DEFAULT 'Untitled',
    model_id TEXT NOT NULL,
    system_prompt TEXT,
    parameters TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS session_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    model_id TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS workspaces (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_runs_model_id ON runs(model_id);
CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at);
CREATE INDEX IF NOT EXISTS idx_runs_experiment_id ON runs(experiment_id);
CREATE INDEX IF NOT EXISTS idx_session_messages_session
    ON session_messages(session_id);
"""

SCHEMA_V2 = """
ALTER TABLE sessions ADD COLUMN workspace_id TEXT REFERENCES workspaces(id);
ALTER TABLE prompt_templates ADD COLUMN workspace_id TEXT REFERENCES workspaces(id);
ALTER TABLE prompt_templates ADD COLUMN category TEXT DEFAULT '';
ALTER TABLE prompt_templates ADD COLUMN created_at TEXT DEFAULT (datetime('now'));
"""

# Whitelist of allowed column names for dynamic model upserts
_ALLOWED_MODEL_COLS = frozenset({
    "quantization",
    "parameter_count",
    "context_length",
    "size_bytes",
    "capabilities",
    "last_used_at",
})


def _gen_id() -> str:
    """Generate a 12-char hex ID (48 bits entropy, no dashes)."""
    return uuid.uuid4().hex[:12]


class Database:
    """Async SQLite wrapper for all persistence."""

    def __init__(self):
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        path = str(db_path())
        self._db = await aiosqlite.connect(path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")
        await self._db.execute("PRAGMA busy_timeout=5000")

        # Schema versioning
        row = await (
            await self._db.execute("PRAGMA user_version")
        ).fetchone()
        version = row[0] if row else 0

        if version < 1:
            await self._db.executescript(SCHEMA_V1)
            await self._db.execute("PRAGMA user_version = 2")
        elif version < 2:
            # Migrate V1 → V2: add workspace support
            for stmt in SCHEMA_V2.strip().split(";"):
                stmt = stmt.strip()
                if stmt:
                    try:
                        await self._db.execute(stmt)
                    except Exception:
                        pass  # Column may already exist
            await self._db.execute("PRAGMA user_version = 2")

        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("Database not connected — call connect() first")
        return self._db

    # ── Runs ─────────────────────────────────────────────────────────────

    async def record_run(
        self,
        model_id: str,
        prompt: str,
        response: str,
        params: dict[str, Any],
        ttft_ms: float | None = None,
        tokens_per_second: float | None = None,
        total_latency_ms: float | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        hw_device: str | None = None,
        hw_gpu_util_avg: float | None = None,
        hw_cpu_util_avg: float | None = None,
        hw_ram_used_gb: float | None = None,
        experiment_id: str | None = None,
        comparison_id: str | None = None,
    ) -> str:
        run_id = _gen_id()
        await self.db.execute(
            """INSERT INTO runs
               (id, model_id, prompt, response, parameters,
                ttft_ms, tokens_per_second, total_latency_ms,
                prompt_tokens, completion_tokens,
                hw_device, hw_gpu_util_avg, hw_cpu_util_avg, hw_ram_used_gb,
                experiment_id, comparison_id)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                run_id,
                model_id,
                prompt,
                response,
                json.dumps(params),
                ttft_ms,
                tokens_per_second,
                total_latency_ms,
                prompt_tokens,
                completion_tokens,
                hw_device,
                hw_gpu_util_avg,
                hw_cpu_util_avg,
                hw_ram_used_gb,
                experiment_id,
                comparison_id,
            ),
        )
        await self.db.commit()
        return run_id

    async def update_run_scores(
        self,
        run_id: str,
        bleu: float | None = None,
        rouge_l: float | None = None,
        llm_judge: float | None = None,
    ) -> None:
        sets = []
        vals: list = []
        if bleu is not None:
            sets.append("score_bleu=?")
            vals.append(bleu)
        if rouge_l is not None:
            sets.append("score_rouge_l=?")
            vals.append(rouge_l)
        if llm_judge is not None:
            sets.append("score_llm_judge=?")
            vals.append(llm_judge)
        if not sets:
            return
        vals.append(run_id)
        await self.db.execute(
            f"UPDATE runs SET {', '.join(sets)} WHERE id=?", vals
        )
        await self.db.commit()

    async def list_runs(
        self, limit: int = 100, model_id: str | None = None
    ) -> list[dict]:
        if model_id:
            cursor = await self.db.execute(
                """SELECT * FROM runs WHERE model_id=?
                   ORDER BY created_at DESC LIMIT ?""",
                (model_id, limit),
            )
        else:
            cursor = await self.db.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
        cols = [d[0] for d in cursor.description]
        rows = await cursor.fetchall()
        return [dict(zip(cols, row)) for row in rows]

    async def get_run(self, run_id: str) -> dict | None:
        cursor = await self.db.execute(
            "SELECT * FROM runs WHERE id=?", (run_id,)
        )
        cols = [d[0] for d in cursor.description]
        row = await cursor.fetchone()
        return dict(zip(cols, row)) if row else None

    # ── Models ───────────────────────────────────────────────────────────

    async def upsert_model(
        self, model_id: str, backend: str, name: str, **kwargs
    ) -> None:
        # Validate column names against whitelist (prevent SQL injection)
        invalid = set(kwargs.keys()) - _ALLOWED_MODEL_COLS
        if invalid:
            raise ValueError(f"Invalid model columns: {invalid}")

        existing = await self.db.execute(
            "SELECT id FROM models WHERE id=?", (model_id,)
        )
        if await existing.fetchone():
            if kwargs:
                sets = ", ".join(f"{k}=?" for k in kwargs)
                vals = list(kwargs.values()) + [model_id]
                await self.db.execute(
                    f"UPDATE models SET {sets} WHERE id=?", vals
                )
        else:
            cols = "id, backend, name"
            placeholders = "?, ?, ?"
            vals: list = [model_id, backend, name]
            if kwargs:
                cols += "," + ",".join(kwargs.keys())
                placeholders += ",?" * len(kwargs)
                vals.extend(kwargs.values())
            await self.db.execute(
                f"INSERT INTO models ({cols}) VALUES ({placeholders})", vals
            )
        await self.db.commit()

    # ── Templates ────────────────────────────────────────────────────────

    async def save_template(
        self, name: str, content: str, variables: list[str]
    ) -> str:
        tid = _gen_id()
        await self.db.execute(
            """INSERT INTO prompt_templates (id, name, content, variables)
               VALUES (?, ?, ?, ?)""",
            (tid, name, content, json.dumps(variables)),
        )
        await self.db.commit()
        return tid

    async def list_templates(self) -> list[dict]:
        cursor = await self.db.execute(
            "SELECT * FROM prompt_templates ORDER BY name"
        )
        cols = [d[0] for d in cursor.description]
        rows = await cursor.fetchall()
        return [dict(zip(cols, row)) for row in rows]

    # ── Sessions ─────────────────────────────────────────────────────────

    async def create_session(
        self,
        model_id: str,
        name: str | None = None,
        params: dict | None = None,
    ) -> str:
        sid = _gen_id()
        display_name = name or "Untitled"
        await self.db.execute(
            """INSERT INTO sessions (id, name, model_id, parameters)
               VALUES (?, ?, ?, ?)""",
            (sid, display_name, model_id, json.dumps(params or {})),
        )
        await self.db.commit()
        return sid

    async def add_session_message(
        self,
        session_id: str,
        role: str,
        content: str,
        model_id: str | None = None,
    ) -> None:
        await self.db.execute(
            """INSERT INTO session_messages
               (session_id, role, content, model_id)
               VALUES (?, ?, ?, ?)""",
            (session_id, role, content, model_id),
        )
        await self.db.execute(
            "UPDATE sessions SET updated_at=datetime('now') WHERE id=?",
            (session_id,),
        )
        await self.db.commit()

    async def update_session_name(
        self, session_id: str, name: str
    ) -> None:
        await self.db.execute(
            "UPDATE sessions SET name=? WHERE id=?", (name, session_id)
        )
        await self.db.commit()

    async def list_sessions(self) -> list[dict]:
        cursor = await self.db.execute(
            """SELECT s.*, COUNT(m.id) as message_count
               FROM sessions s
               LEFT JOIN session_messages m ON m.session_id = s.id
               GROUP BY s.id
               ORDER BY s.updated_at DESC"""
        )
        cols = [d[0] for d in cursor.description]
        rows = await cursor.fetchall()
        return [dict(zip(cols, row)) for row in rows]

    async def get_session(self, session_id: str) -> dict | None:
        cursor = await self.db.execute(
            "SELECT * FROM sessions WHERE id=?", (session_id,)
        )
        cols = [d[0] for d in cursor.description]
        row = await cursor.fetchone()
        return dict(zip(cols, row)) if row else None

    async def get_session_messages(
        self, session_id: str, limit: int = 1000
    ) -> list[dict]:
        cursor = await self.db.execute(
            """SELECT role, content, model_id, created_at
               FROM session_messages
               WHERE session_id=?
               ORDER BY id ASC LIMIT ?""",
            (session_id, limit),
        )
        cols = [d[0] for d in cursor.description]
        rows = await cursor.fetchall()
        return [dict(zip(cols, row)) for row in rows]

    # ── Workspaces ─────────────────────────────────────────────────────

    async def create_workspace(self, name: str, description: str = "") -> str:
        wid = _gen_id()
        await self.db.execute(
            "INSERT INTO workspaces (id, name, description) VALUES (?, ?, ?)",
            (wid, name, description),
        )
        await self.db.commit()
        return wid

    async def list_workspaces(self) -> list[dict]:
        cursor = await self.db.execute(
            "SELECT * FROM workspaces ORDER BY created_at DESC"
        )
        cols = [d[0] for d in cursor.description]
        rows = await cursor.fetchall()
        return [dict(zip(cols, row)) for row in rows]

    async def delete_workspace(self, workspace_id: str) -> None:
        await self.db.execute(
            "UPDATE sessions SET workspace_id=NULL WHERE workspace_id=?",
            (workspace_id,),
        )
        await self.db.execute(
            "DELETE FROM workspaces WHERE id=?", (workspace_id,)
        )
        await self.db.commit()

    # ── Enhanced Templates ────────────────────────────────────────────

    async def get_template(self, template_id: str) -> dict | None:
        cursor = await self.db.execute(
            "SELECT * FROM prompt_templates WHERE id=?", (template_id,)
        )
        cols = [d[0] for d in cursor.description]
        row = await cursor.fetchone()
        return dict(zip(cols, row)) if row else None

    async def update_template(
        self, template_id: str, name: str, content: str, variables: list[str]
    ) -> None:
        await self.db.execute(
            "UPDATE prompt_templates SET name=?, content=?, variables=? WHERE id=?",
            (name, content, json.dumps(variables), template_id),
        )
        await self.db.commit()

    async def delete_template(self, template_id: str) -> None:
        await self.db.execute(
            "DELETE FROM prompt_templates WHERE id=?", (template_id,)
        )
        await self.db.commit()

    async def fork_session(
        self, session_id: str, from_message_index: int
    ) -> str:
        """Fork a session from a specific message point."""
        session = await self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        messages = await self.get_session_messages(session_id)
        forked_messages = messages[:from_message_index + 1]

        new_id = await self.create_session(
            model_id=session["model_id"],
            name=f"Fork of {session['name']}",
            params=json.loads(session.get("parameters", "{}")) if session.get("parameters") else {},
        )
        for msg in forked_messages:
            await self.add_session_message(
                new_id, msg["role"], msg["content"], msg.get("model_id")
            )
        return new_id

    async def delete_session(self, session_id: str) -> None:
        await self.db.execute(
            "DELETE FROM session_messages WHERE session_id=?",
            (session_id,),
        )
        await self.db.execute(
            "DELETE FROM sessions WHERE id=?", (session_id,)
        )
        await self.db.commit()
