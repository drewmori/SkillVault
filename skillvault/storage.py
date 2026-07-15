"""Small local persistence layer for approved SkillVault knowledge."""

from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path

from .data import ExpertCase, infer_features


def _text(value: object) -> str:
    return str(value or "").strip()


def case_to_dict(case: ExpertCase) -> dict[str, object]:
    media = []
    for item in case.media:
        raw = item.get("bytes", b"")
        if isinstance(raw, bytes):
            encoded = base64.b64encode(raw).decode("ascii")
        else:
            encoded = ""
        media.append({"name": _text(item.get("name")), "type": _text(item.get("type")), "bytes": encoded})
    return {
        "summary": case.summary,
        "label": case.label,
        "reasoning": case.reasoning,
        "features": case.features,
        "source_file": case.source_file,
        "instructions": case.instructions,
        "software": case.software,
        "methods": case.methods,
        "outcome": case.outcome,
        "media": media,
    }


def case_from_dict(item: dict[str, object]) -> ExpertCase:
    media = []
    for media_item in item.get("media", []) or []:
        if not isinstance(media_item, dict):
            continue
        encoded = _text(media_item.get("bytes"))
        try:
            raw = base64.b64decode(encoded) if encoded else b""
        except ValueError:
            raw = b""
        media.append({"name": _text(media_item.get("name")), "type": _text(media_item.get("type")), "bytes": raw})
    summary = _text(item.get("summary"))
    return ExpertCase(
        summary,
        _text(item.get("label")),
        _text(item.get("reasoning")),
        item.get("features") if isinstance(item.get("features"), dict) else infer_features(summary),
        _text(item.get("source_file")) or "approved_expert_decision.md",
        _text(item.get("instructions")),
        _text(item.get("software")),
        _text(item.get("methods")),
        _text(item.get("outcome")),
        media,
    )


def load_cases(path: Path) -> list[ExpertCase]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    return [case_from_dict(item) for item in payload if isinstance(item, dict)]


def save_cases(path: Path, cases: list[ExpertCase]) -> None:
    path.write_text(json.dumps([case_to_dict(case) for case in cases], indent=2), encoding="utf-8")


def append_case(path: Path, case: ExpertCase) -> None:
    cases = load_cases(path)
    cases.append(case)
    save_cases(path, cases)


def save_approved_plan(path: Path, query: str, label: str, confidence: float, sources: list[str], answer: str) -> None:
    history = []
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            history = payload if isinstance(payload, list) else []
        except (OSError, json.JSONDecodeError):
            history = []
    history.append({
        "approved_at": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "label": label,
        "confidence": confidence,
        "sources": sources,
        "answer": answer,
    })
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")
