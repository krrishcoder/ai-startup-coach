from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional, Sequence


@dataclass(frozen=True)
class Message:
    user_id: str
    conversation_id: str
    role: str  # 'user' | 'assistant' | 'system'
    content: str
    timestamp: str


def _connect(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_db(db_path: str) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_messages_user_conv
            ON messages(user_id, conversation_id, timestamp);
            """
        )
        conn.commit()


def save_message(
    db_path: str,
    *,
    user_id: str,
    conversation_id: str,
    role: str,
    content: str,
    timestamp: Optional[str] = None,
) -> None:
    ensure_db(db_path)
    ts = timestamp or datetime.utcnow().isoformat(timespec="seconds") + "Z"
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO messages(user_id, conversation_id, role, content, timestamp)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, conversation_id, role, content, ts),
        )
        conn.commit()


def save_messages(db_path: str, messages: Sequence[Message]) -> None:
    if not messages:
        return
    ensure_db(db_path)
    with _connect(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO messages(user_id, conversation_id, role, content, timestamp)
            VALUES (?, ?, ?, ?, ?)
            """,
            [(m.user_id, m.conversation_id, m.role, m.content, m.timestamp) for m in messages],
        )
        conn.commit()


def load_messages(
    db_path: str,
    *,
    user_id: str,
    conversation_id: str,
    limit: int = 50,
) -> list[Message]:
    ensure_db(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT user_id, conversation_id, role, content, timestamp
            FROM messages
            WHERE user_id = ? AND conversation_id = ?
            ORDER BY timestamp ASC
            LIMIT ?;
            """,
            (user_id, conversation_id, limit),
        ).fetchall()

    return [
        Message(
            user_id=row["user_id"],
            conversation_id=row["conversation_id"],
            role=row["role"],
            content=row["content"],
            timestamp=row["timestamp"],
        )
        for row in rows
    ]
