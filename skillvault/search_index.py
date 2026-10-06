"""Persistent, local retrieval index for large SkillVault knowledge bases."""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json
import re
import sqlite3
from threading import RLock

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .data import ExpertCase


_INDEX_LOCK = RLock()
_QUERY_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do",
    "for", "from", "had", "has", "have", "how", "i", "if", "in", "is", "it",
    "me", "my", "of", "on", "or", "our", "should", "that", "the", "this", "to",
    "was", "we", "were", "what", "when", "where", "which", "with", "would",
}


@dataclass(frozen=True)
class IndexStats:
    backend: str
    indexed_count: int
    fingerprint: str
    rebuilt: bool


def _case_text(case: ExpertCase) -> str:
    return " ".join(
        value.strip()
        for value in (
            case.summary,
            case.reasoning,
            case.instructions,
            case.software,
            case.methods,
            case.outcome,
            case.source_file,
            case.expert_owner,
            case.department,
            case.goal,
            case.chosen_approach,
            case.alternatives_considered,
            case.constraints,
            case.reusable_rule,
            case.exceptions,
        )
        if value and value.strip()
    )


def _fingerprint(cases: list[ExpertCase]) -> str:
    payload = [
        {
            "summary": case.summary,
            "label": case.label,
            "reasoning": case.reasoning,
            "instructions": case.instructions,
            "software": case.software,
            "methods": case.methods,
            "outcome": case.outcome,
            "source_file": case.source_file,
            "features": case.features,
            "expert_owner": case.expert_owner,
            "department": case.department,
            "approval_date": case.approval_date,
            "last_reviewed_date": case.last_reviewed_date,
            "expiration_date": case.expiration_date,
            "lifecycle_status": case.lifecycle_status,
            "replaces_source": case.replaces_source,
            "goal": case.goal,
            "chosen_approach": case.chosen_approach,
            "alternatives_considered": case.alternatives_considered,
            "constraints": case.constraints,
            "reusable_rule": case.reusable_rule,
            "exceptions": case.exceptions,
        }
        for case in cases
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return sha256(encoded).hexdigest()


class PersistentKnowledgeIndex:
    """Persist vectors and use SQLite FTS5 to shortlist retrieval candidates."""

    def __init__(self, base_path: Path) -> None:
        self.base_path = Path(base_path)
        self.sqlite_path = self.base_path.with_suffix(".sqlite3")
        self.vector_path = self.base_path.with_suffix(".joblib")
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.fts_enabled = False
        self.indexed_count = 0
        self.fingerprint = ""

    def sync(self, cases: list[ExpertCase]) -> tuple[TfidfVectorizer, object, IndexStats]:
        """Load a matching vector index or rebuild it when approved knowledge changes."""
        fingerprint = _fingerprint(cases)
        rebuilt = False
        bundle = None

        with _INDEX_LOCK:
            if self.vector_path.exists():
                try:
                    loaded = joblib.load(self.vector_path)
                    if isinstance(loaded, dict) and loaded.get("fingerprint") == fingerprint and loaded.get("count") == len(cases):
                        bundle = loaded
                except (OSError, EOFError, ValueError, TypeError, KeyError, AttributeError):
                    bundle = None

            if bundle is None:
                texts = [_case_text(case) for case in cases]
                vectorizer = TfidfVectorizer(
                    stop_words="english",
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    dtype=np.float32,
                    max_features=150_000,
                )
                matrix = vectorizer.fit_transform(texts)
                bundle = {
                    "fingerprint": fingerprint,
                    "count": len(cases),
                    "vectorizer": vectorizer,
                    "matrix": matrix,
                }
                temporary = self.vector_path.with_suffix(".joblib.tmp")
                joblib.dump(bundle, temporary, compress=3)
                temporary.replace(self.vector_path)
                rebuilt = True

            self.fts_enabled = self._sync_fts(cases, fingerprint)

        self.indexed_count = len(cases)
        self.fingerprint = fingerprint
        backend = "SQLite FTS5 + persisted TF-IDF vectors" if self.fts_enabled else "Persisted TF-IDF vectors"
        stats = IndexStats(backend, len(cases), fingerprint, rebuilt)
        return bundle["vectorizer"], bundle["matrix"], stats

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.sqlite_path, timeout=15)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        return connection

    def _sync_fts(self, cases: list[ExpertCase], fingerprint: str) -> bool:
        try:
            with closing(self._connect()) as connection, connection:
                connection.execute("CREATE TABLE IF NOT EXISTS index_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
                connection.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5("
                    "case_id UNINDEXED, content, tokenize='porter unicode61')"
                )
                current = connection.execute(
                    "SELECT value FROM index_meta WHERE key = 'fingerprint'"
                ).fetchone()
                if current is None or current[0] != fingerprint:
                    connection.execute("DELETE FROM knowledge_fts")
                    connection.executemany(
                        "INSERT INTO knowledge_fts(case_id, content) VALUES (?, ?)",
                        ((index, _case_text(case)) for index, case in enumerate(cases)),
                    )
                    connection.execute(
                        "INSERT INTO index_meta(key, value) VALUES ('fingerprint', ?) "
                        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                        (fingerprint,),
                    )
                    connection.execute(
                        "INSERT INTO index_meta(key, value) VALUES ('count', ?) "
                        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                        (str(len(cases)),),
                    )
            return True
        except sqlite3.Error:
            return False

    @staticmethod
    def _fts_query(text: str) -> str:
        tokens = []
        for token in re.findall(r"[A-Za-z0-9_]{2,}", text.lower()):
            if token not in _QUERY_STOP_WORDS and token not in tokens:
                tokens.append(token)
            if len(tokens) == 24:
                break
        return " OR ".join(f'"{token}"' for token in tokens)

    def candidate_indices(
        self,
        query: str,
        allowed_indices: list[int] | None = None,
        limit: int = 256,
    ) -> list[int]:
        """Return a fast lexical shortlist; callers rerank it with vector similarity."""
        allowed = set(allowed_indices) if allowed_indices is not None else None
        universe = allowed_indices if allowed_indices is not None else list(range(self.indexed_count))
        if len(universe) <= limit or not self.fts_enabled:
            return list(universe)

        fts_query = self._fts_query(query)
        if not fts_query:
            return list(universe)

        fetch_limit = min(max(limit * 6, limit), 3000)
        try:
            with closing(self._connect()) as connection, connection:
                rows = connection.execute(
                    "SELECT case_id FROM knowledge_fts WHERE knowledge_fts MATCH ? "
                    "ORDER BY bm25(knowledge_fts) LIMIT ?",
                    (fts_query, fetch_limit),
                ).fetchall()
        except sqlite3.Error:
            return list(universe)

        candidates = [int(row[0]) for row in rows if allowed is None or int(row[0]) in allowed]
        return candidates[:limit] or list(universe)
