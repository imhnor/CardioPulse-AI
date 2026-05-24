"""
upload_handler.py
-----------------
Manages temporary ECG file storage during a request lifecycle.

Handles:
  - Single-file uploads (XML, DICOM, SCP-ECG)
  - Dual-file WFDB uploads (.hea + .dat stored together)
  - Session-based temp directories (auto-cleaned after prediction)
"""

import os
import uuid
import shutil
import tempfile
from typing import Optional


# Temporary base directory inside the app folder
_TMP_BASE = os.path.join(os.path.dirname(__file__), "..", "tmp_uploads")


def _ensure_tmp_base():
    os.makedirs(_TMP_BASE, exist_ok=True)


class UploadSession:
    """
    Represents a single upload session with a unique ID.
    Manages temp files and provides path resolution.
    """

    def __init__(self, session_id: str, directory: str):
        self.session_id  = session_id
        self.directory   = directory
        self.files: list = []   # list of (original_filename, saved_path)

    def add_file(self, original_name: str, content: bytes) -> str:
        """Save file bytes to session directory. Returns the saved path."""
        safe_name  = os.path.basename(original_name)
        dest_path  = os.path.join(self.directory, safe_name)
        with open(dest_path, "wb") as f:
            f.write(content)
        self.files.append((original_name, dest_path))
        return dest_path

    def get_primary_file(self) -> Optional[str]:
        if not self.files:
            return None

        for name, path in self.files:
            if name.lower().endswith(".hea"):
                return os.path.splitext(path)[0]  # return base record path

        # For single file uploads (XML, DICOM, SCP-ECG, etc.)
        return self.files[0][1]

    def cleanup(self):
        """Delete all session files and directory."""
        try:
            shutil.rmtree(self.directory, ignore_errors=True)
        except Exception:
            pass


# ── In-memory session store ───────────────────────────────────────────────────
_sessions: dict[str, UploadSession] = {}


def create_session() -> UploadSession:
    """Create a new upload session with a unique temp directory."""
    _ensure_tmp_base()
    session_id = str(uuid.uuid4())
    session_dir = os.path.join(_TMP_BASE, session_id)
    os.makedirs(session_dir, exist_ok=True)
    session = UploadSession(session_id, session_dir)
    _sessions[session_id] = session
    return session


def get_session(session_id: str) -> Optional[UploadSession]:
    """Retrieve an existing session by ID."""
    return _sessions.get(session_id)


def cleanup_session(session_id: str):
    """Remove session from memory and delete temp files."""
    session = _sessions.pop(session_id, None)
    if session:
        session.cleanup()


def cleanup_all():
    """Emergency cleanup of all sessions (e.g., on server shutdown)."""
    for session in list(_sessions.values()):
        session.cleanup()
    _sessions.clear()
    try:
        shutil.rmtree(_TMP_BASE, ignore_errors=True)
    except Exception:
        pass
