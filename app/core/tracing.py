"""
Structured tracing for agent runs. Each run gets a run_id; every
thought -> action -> observation step within that run is stored as a
span with its own sequence number, so a full run can be reconstructed
and replayed in order.

SQLite here (matching the pattern from Projects 2 and 7) keeps this
dependency-light. The docstring at the bottom shows exactly where to
swap in OpenTelemetry or Langfuse if you want a real tracing backend --
the span shape below (name, input, output, metadata, timestamps) maps
directly onto either.
"""
import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field

DB_PATH = os.environ.get("TRACING_DB", "agent_traces.db")


@dataclass
class Span:
    run_id: str
    step_index: int
    span_type: str          # "thought" | "tool_call" | "tool_result" | "guardrail_block" | "final_answer"
    name: str                # e.g. tool name, or "reasoning"
    input_data: dict
    output_data: dict
    metadata: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@contextmanager
def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                task TEXT,
                status TEXT,           -- "running" | "completed" | "halted" | "budget_exceeded"
                halt_reason TEXT,
                total_tokens INTEGER DEFAULT 0,
                total_cost_usd REAL DEFAULT 0,
                step_count INTEGER DEFAULT 0,
                started_at REAL,
                ended_at REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS spans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                step_index INTEGER,
                span_type TEXT,
                name TEXT,
                input_data TEXT,
                output_data TEXT,
                metadata TEXT,
                timestamp REAL
            )
        """)


def start_run(task: str) -> str:
    run_id = str(uuid.uuid4())
    with _conn() as conn:
        conn.execute(
            """INSERT INTO runs (run_id, task, status, started_at) VALUES (?, ?, ?, ?)""",
            (run_id, task, "running", time.time()),
        )
    return run_id


def record_span(span: Span) -> None:
    with _conn() as conn:
        conn.execute(
            """INSERT INTO spans (run_id, step_index, span_type, name, input_data, output_data,
                                   metadata, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (span.run_id, span.step_index, span.span_type, span.name,
             json.dumps(span.input_data), json.dumps(span.output_data),
             json.dumps(span.metadata), span.timestamp),
        )
        conn.execute("UPDATE runs SET step_count = step_count + 1 WHERE run_id = ?", (span.run_id,))


def end_run(run_id: str, status: str, halt_reason: str = "",
            total_tokens: int = 0, total_cost_usd: float = 0.0) -> None:
    with _conn() as conn:
        conn.execute(
            """UPDATE runs SET status = ?, halt_reason = ?, total_tokens = ?,
                                total_cost_usd = ?, ended_at = ? WHERE run_id = ?""",
            (status, halt_reason, total_tokens, total_cost_usd, time.time(), run_id),
        )


def get_run(run_id: str) -> dict | None:
    with _conn() as conn:
        row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    return dict(row) if row else None


def get_spans(run_id: str) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM spans WHERE run_id = ? ORDER BY step_index ASC", (run_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def recent_runs(limit: int = 50) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


# --- OpenTelemetry / Langfuse integration point ---------------------------
# To export these spans to a real tracing backend instead of (or in
# addition to) SQLite:
#
#   from opentelemetry import trace
#   tracer = trace.get_tracer("agentic-guardrails")
#   with tracer.start_as_current_span(span.name) as otel_span:
#       otel_span.set_attribute("run_id", span.run_id)
#       otel_span.set_attribute("input", json.dumps(span.input_data))
#       otel_span.set_attribute("output", json.dumps(span.output_data))
#
# Or for Langfuse: langfuse.span(trace_id=run_id, name=span.name,
#   input=span.input_data, output=span.output_data, metadata=span.metadata)
#
# Call this from record_span() above, alongside (or instead of) the SQLite
# insert, once you've picked a backend.
