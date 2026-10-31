"""
Upload handling for the chatbot, kept free of Streamlit and LangChain so it can be tested.
"""

import hashlib
import uuid
from collections.abc import Iterable
from pathlib import PurePosixPath


def safe_filename(name: str) -> str:
    """Drop directory parts from an uploaded file's name."""
    return PurePosixPath(name.replace("\\", "/")).name or "upload.pdf"


def uploads_fingerprint(files: Iterable[tuple[str, bytes]]) -> str:
    """Identify a set of uploads by content, in any order."""
    digests = sorted(
        f"{name}:{hashlib.sha256(content).hexdigest()}" for name, content in files
    )
    return hashlib.sha256("|".join(digests).encode("utf-8")).hexdigest()


def private_collection_name() -> str:
    """Return a collection name unique to one index, so sessions cannot share documents."""
    return f"session_{uuid.uuid4().hex}"
