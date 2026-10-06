"""Company-scoped, content-addressed storage for original work files."""

from hashlib import sha256
from pathlib import Path
import re

from .importer import WorkFile


def store_artifacts(root: Path, files: list[WorkFile]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for file in files:
        digest = sha256(file.data).hexdigest()
        target = root / digest
        if not target.exists():
            target.write_bytes(file.data)


def read_artifact(root: Path, digest: str) -> bytes | None:
    if not re.fullmatch(r"[a-f0-9]{64}", digest):
        return None
    root = root.resolve()
    target = (root / digest).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        return None
    content = target.read_bytes()
    return content if sha256(content).hexdigest() == digest else None
