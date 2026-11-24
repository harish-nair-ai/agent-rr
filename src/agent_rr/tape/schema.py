"""Canonical .tape SQLite DDL (contracts §6, D10) and connection setup (contracts §0)."""

import sqlite3
from pathlib import Path

SCHEMA_VERSION = "0.1"

DDL = """
CREATE TABLE IF NOT EXISTS tape_meta (
    tape_id TEXT PRIMARY KEY,
    schema_version TEXT NOT NULL DEFAULT '0.1',
    created_at REAL NOT NULL,
    command_argv TEXT NOT NULL,
    repo_root TEXT NOT NULL,
    base_commit_sha TEXT NOT NULL,
    parent_tape_id TEXT,
    fork_step_idx INTEGER,
    chaos_seed INTEGER,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS steps (
    step_idx INTEGER PRIMARY KEY,
    ts_start REAL NOT NULL,
    ts_end REAL NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('llm_call','tool_call','chaos_fault')),
    pre_tree_sha TEXT NOT NULL,
    post_tree_sha TEXT NOT NULL,
    diff_text TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'ok'
        CHECK (status IN ('ok','error','steered','injected_fault'))
);
CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    step_idx INTEGER NOT NULL REFERENCES steps(step_idx),
    provider TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    model TEXT NOT NULL,
    request_json TEXT NOT NULL,
    response_json TEXT NOT NULL,
    raw_sse_chunks TEXT,
    http_status INTEGER NOT NULL,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    latency_ms REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    step_idx INTEGER NOT NULL REFERENCES steps(step_idx),
    tool_name TEXT NOT NULL,
    arguments_json TEXT NOT NULL,
    result_text TEXT NOT NULL,
    is_error INTEGER NOT NULL DEFAULT 0,
    exit_code INTEGER
);
CREATE TABLE IF NOT EXISTS chaos_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    step_idx INTEGER NOT NULL REFERENCES steps(step_idx),
    preset TEXT NOT NULL,
    seed INTEGER NOT NULL,
    target TEXT NOT NULL,
    detail_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_llm_calls_step ON llm_calls(step_idx);
CREATE INDEX IF NOT EXISTS idx_tool_calls_step ON tool_calls(step_idx);
CREATE INDEX IF NOT EXISTS idx_chaos_events_step ON chaos_events(step_idx);
"""


def connect(path: Path) -> sqlite3.Connection:
    """Open a tape connection with the contract PRAGMAs; transactions are explicit."""
    conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn
