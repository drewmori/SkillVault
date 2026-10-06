"""Small local persistence layer for approved SkillVault knowledge."""

from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

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
        "capture_evidence": case.capture_evidence,
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
        _text(item.get("expert_owner")) or "Legacy knowledge owner",
        _text(item.get("department")) or "Unassigned",
        _text(item.get("approval_date")) or "2026-07-13",
        _text(item.get("last_reviewed_date")) or _text(item.get("approval_date")) or "2026-07-13",
        _text(item.get("expiration_date")) or "2027-07-13",
        _text(item.get("lifecycle_status")) or "Current",
        _text(item.get("replaces_source")),
        _text(item.get("goal")),
        _text(item.get("chosen_approach")),
        _text(item.get("alternatives_considered")),
        _text(item.get("constraints")),
        _text(item.get("reusable_rule")),
        _text(item.get("exceptions")),
        item.get("capture_evidence") if isinstance(item.get("capture_evidence"), dict) else {},
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
    if case.replaces_source:
        replacement_target = case.replaces_source.strip().lower()
        for existing in cases:
            if existing.source_file.strip().lower() == replacement_target:
                existing.lifecycle_status = "Replaced"
    cases.append(case)
    save_cases(path, cases)


def save_approved_plan(
    path: Path,
    query: str,
    label: str,
    confidence: float,
    sources: list[str],
    answer: str,
    lineage: dict[str, object] | None = None,
) -> str:
    history = []
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            history = payload if isinstance(payload, list) else []
        except (OSError, json.JSONDecodeError):
            history = []
    plan_id = f"plan-{uuid4().hex[:12]}"
    stored_lineage = dict(lineage or {})
    if stored_lineage:
        stored_lineage["plan_id"] = plan_id
    history.append({
        "plan_id": plan_id,
        "approved_at": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "label": label,
        "confidence": confidence,
        "sources": sources,
        "answer": answer,
        "decision_lineage": stored_lineage,
    })
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return plan_id


def load_approved_plans(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    history = [item for item in payload if isinstance(item, dict)] if isinstance(payload, list) else []
    migrated = False
    for index, plan in enumerate(history, start=1):
        if not _text(plan.get("plan_id")):
            plan_id = f"legacy-plan-{index:04d}"
            plan["plan_id"] = plan_id
            lineage = plan.get("decision_lineage")
            if isinstance(lineage, dict):
                lineage["plan_id"] = plan_id
            migrated = True
    if migrated:
        path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return history


def load_recommendation_feedback(path: Path) -> list[dict[str, object]]:
    """Load the company-scoped audit log of recommendation feedback."""
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return [item for item in payload if isinstance(item, dict)] if isinstance(payload, list) else []


def save_recommendation_feedback(path: Path, feedback: dict[str, object]) -> str:
    """Append one feedback event without treating it as approved knowledge."""
    history = load_recommendation_feedback(path)
    feedback_id = f"feedback-{uuid4().hex[:12]}"
    record = dict(feedback)
    record.update(
        {
            "feedback_id": feedback_id,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    history.append(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return feedback_id


def save_plan_outcome(
    path: Path,
    plan_id: str,
    effectiveness: str,
    steps_changed: str,
    actual_outcome: str,
    promote_to_knowledge: bool,
    promoted_source: str = "",
) -> dict[str, object]:
    """Attach real-world follow-up to an approved plan and its lineage."""
    history = load_approved_plans(path)
    updated: dict[str, object] | None = None
    recorded_at = datetime.now(timezone.utc).isoformat()
    for plan in history:
        if _text(plan.get("plan_id")) != plan_id:
            continue
        feedback = {
            "recorded_at": recorded_at,
            "effectiveness": effectiveness,
            "steps_changed": steps_changed,
            "actual_outcome": actual_outcome,
            "promoted_to_knowledge": promote_to_knowledge,
            "promoted_source": promoted_source,
        }
        plan["outcome_feedback"] = feedback
        lineage = plan.get("decision_lineage")
        if not isinstance(lineage, dict):
            lineage = {}
            plan["decision_lineage"] = lineage
        lineage["outcome"] = {
            "status": "Recorded",
            "effectiveness": effectiveness,
            "summary": actual_outcome,
            "steps_changed": steps_changed,
            "recorded_at": recorded_at,
        }
        updated = plan
        break
    if updated is None:
        raise KeyError(f"Approved plan not found: {plan_id}")
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return updated
