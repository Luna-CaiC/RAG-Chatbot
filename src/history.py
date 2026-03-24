"""
src/history.py — Persistent Chat History Manager
==================================================
Saves and loads chat sessions to/from a local JSON file
so conversations survive page refreshes.
"""

import json
import os
import uuid
from datetime import datetime

HISTORY_FILE = os.path.join("data", "chat_sessions.json")


class ChatHistoryManager:
    """Manage persistent chat sessions stored as a local JSON file."""

    def __init__(self, filepath: str = HISTORY_FILE):
        self.filepath = filepath
        self._ensure_file()

    def _ensure_file(self):
        """Create the history file if it doesn't exist."""
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        if not os.path.exists(self.filepath):
            self._write([])

    def _read(self) -> list:
        """Read all sessions from disk."""
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _write(self, sessions: list):
        """Write all sessions to disk."""
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(sessions, f, ensure_ascii=False, indent=2)

    def create_session(self, doc_names: list[str]) -> str:
        """Create a new session and return its ID."""
        session_id = uuid.uuid4().hex[:8]
        sessions = self._read()
        sessions.insert(0, {
            "id": session_id,
            "timestamp": datetime.now().isoformat(),
            "documents": doc_names,
            "messages": [],
        })
        # Keep only the latest 50 sessions
        self._write(sessions[:50])
        return session_id

    def save_messages(self, session_id: str, messages: list[dict]):
        """Update messages for a given session."""
        sessions = self._read()
        for s in sessions:
            if s["id"] == session_id:
                s["messages"] = messages
                break
        self._write(sessions)

    def load_sessions(self) -> list[dict]:
        """Return all saved sessions (summary only, no messages)."""
        sessions = self._read()
        return [
            {
                "id": s["id"],
                "timestamp": s.get("timestamp", ""),
                "documents": s.get("documents", []),
                "message_count": len(s.get("messages", [])),
            }
            for s in sessions
        ]

    def get_session(self, session_id: str) -> dict | None:
        """Return full session data including messages."""
        sessions = self._read()
        for s in sessions:
            if s["id"] == session_id:
                return s
        return None

    def delete_session(self, session_id: str):
        """Remove a session by ID."""
        sessions = self._read()
        sessions = [s for s in sessions if s["id"] != session_id]
        self._write(sessions)
