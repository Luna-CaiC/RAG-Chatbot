"""
src/history.py — Persistent Chat History Manager
==================================================
Saves and loads chat sessions to/from a local JSON file.
Also manages per-session file storage so sessions can be resumed.
"""

import json
import os
import shutil
import uuid
from datetime import datetime

HISTORY_FILE  = os.path.join("data", "chat_sessions.json")
FILES_BASE_DIR = os.path.join("data", "session_files")


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

    # ── Session CRUD ──────────────────────────────────────────────────

    def create_session(self, session_id: str, doc_names: list[str]) -> str:
        """Create a new session with a provided ID and return it."""
        sessions = self._read()
        sessions.insert(0, {
            "id": session_id,
            "timestamp": datetime.now().isoformat(),
            "documents": doc_names,
            "messages": [],
        })
        self._write(sessions[:50])  # Keep latest 50
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
        """Remove a session from the list AND delete its stored files and vectors."""
        # Remove from JSON
        sessions = self._read()
        sessions = [s for s in sessions if s["id"] != session_id]
        self._write(sessions)
        # Delete stored raw files
        self.delete_session_files(session_id)
        # Delete persisted Chroma vector store
        vector_dir = os.path.join("data", "session_vectors", session_id)
        if os.path.exists(vector_dir):
            shutil.rmtree(vector_dir)

    # ── Session file storage ──────────────────────────────────────────

    def save_session_files(self, session_id: str, file_data: list[tuple[str, bytes]]):
        """
        Persist uploaded file bytes to disk so the session can be resumed later.
        Args:
            session_id: The session identifier.
            file_data: List of (filename, bytes) tuples.
        """
        session_dir = os.path.join(FILES_BASE_DIR, session_id)
        os.makedirs(session_dir, exist_ok=True)
        for name, data in file_data:
            # Sanitize filename to avoid path traversal issues
            safe_name = os.path.basename(name).replace("/", "_")
            with open(os.path.join(session_dir, safe_name), "wb") as f:
                f.write(data)

    def get_session_files_dir(self, session_id: str) -> str:
        """Return the directory where session files are stored."""
        return os.path.join(FILES_BASE_DIR, session_id)

    def session_has_stored_files(self, session_id: str) -> bool:
        """Return True if the session has stored files on disk."""
        d = self.get_session_files_dir(session_id)
        return os.path.isdir(d) and bool(os.listdir(d))

    def delete_session_files(self, session_id: str):
        """Delete all stored files for a session."""
        session_dir = self.get_session_files_dir(session_id)
        if os.path.exists(session_dir):
            shutil.rmtree(session_dir)
