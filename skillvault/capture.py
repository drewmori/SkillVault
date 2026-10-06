"""Evidence comparison and a short, explicit employee decision interview."""

from dataclasses import replace
from difflib import unified_diff
from hashlib import sha256

from .importer import DecisionDraft, WorkFile, draft_from_completed_work, extract_text


QUESTIONS = (
    ("goal", "What were you trying to achieve or fix?"),
    ("chosen_approach", "What specific change or approach did you choose?"),
    ("reasoning", "Why did you choose that approach? What made it fit this situation?"),
    ("instructions", "What steps did you actually take that another employee could repeat?"),
    ("outcome", "What happened afterward, and how did you check it? If not measured yet, say so."),
    ("alternatives_considered", "What other approach did you consider, and why did you reject it? Say none if there wasn't one."),
    ("constraints", "What limits mattered: time, budget, systems, audience, or company rules?"),
    ("exceptions", "When would this approach be unsuitable or need a different decision?"),
)


def interview_questions(draft: DecisionDraft, limit: int = 3) -> list[tuple[str, str]]:
    """Ask for missing fields; never present an inferred motive as fact."""
    placeholders = ("must add", "must record", "replace these draft steps")
    priority = {"reasoning": 0, "instructions": 1, "outcome": 2, "chosen_approach": 3, "goal": 4}
    return [
        (field, question)
        for field, question in sorted(QUESTIONS, key=lambda item: priority.get(item[0], 10))
        if not str(getattr(draft, field)).strip()
        or any(marker in str(getattr(draft, field)).lower() for marker in placeholders)
    ][:limit]


def next_interview_question(draft: DecisionDraft) -> tuple[str, str] | None:
    """One high-value gap; reference the observed action without claiming its motive."""
    missing = interview_questions(draft, limit=len(QUESTIONS))
    if not missing:
        return None
    field, question = missing[0]
    if field == "reasoning" and draft.chosen_approach.strip():
        choice = draft.chosen_approach.strip().splitlines()[0][:160]
        question = f"The work shows this choice: {choice}. Why was it chosen?"
    elif field == "outcome":
        question = "What happened after this work was used? If you have not measured it yet, say ‘Not measured yet’."
    return field, question


def apply_interview(draft: DecisionDraft, answers: dict[str, str]) -> DecisionDraft:
    allowed = dict(QUESTIONS)
    updates = {key: value.strip() for key, value in answers.items() if key in allowed and value.strip()}
    evidence = dict(draft.capture_evidence)
    confirmed = dict(evidence.get("employee_confirmed", {}))
    confirmed.update(updates)
    evidence["employee_confirmed"] = confirmed
    revised = replace(draft, **updates, capture_evidence=evidence)
    revised.missing_context = [question for _, question in interview_questions(revised, limit=len(QUESTIONS))]
    return revised


def capture_work(
    before: list[WorkFile],
    after: list[WorkFile],
    supporting: list[WorkFile],
    notes: str = "",
    context: dict[str, str] | None = None,
) -> DecisionDraft:
    """Compare bounded text evidence, preserving roles, fingerprints, and excerpts.

    Text differences demonstrate edits, not the employee's intent or success.
    Source line numbers refer to extracted text, not original document layout.
    """
    documents: list[dict[str, object]] = []
    warnings: list[str] = []
    role_text: dict[str, list[str]] = {"before": [], "after": [], "supporting": []}
    for role, files in (("before", before), ("after", after), ("supporting", supporting)):
        for index, work_file in enumerate(files, 1):
            extracted, warning = extract_text(work_file)
            if warning:
                warnings.append(warning)
            excerpt = extracted[:20000]
            if len(extracted) > len(excerpt):
                warnings.append(f"{work_file.name}: comparison uses the first 20,000 extracted characters.")
            if not excerpt.strip():
                warnings.append(f"{work_file.name}: no readable text; image-only or video changes need an employee description.")
            source_id = f"{role}-{index}"
            documents.append({
                "source_id": source_id, "role": role, "filename": work_file.name,
                "sha256": sha256(work_file.data).hexdigest(), "text": excerpt,
                "locator": "Extracted text lines; slide labels are retained for PowerPoint",
            })
            role_text[role].append(f"SOURCE {source_id}: {work_file.name}\n{excerpt}")

    # Extract the final result and supporting trail. Old-version instructions
    # must not become the recommended process just because they appeared first.
    draft = draft_from_completed_work(after + supporting, notes, context)
    comparison = ""
    if before and after:
        comparison = "\n".join(unified_diff(
            "\n\n".join(role_text["before"]).splitlines(),
            "\n\n".join(role_text["after"]).splitlines(),
            fromfile="BEFORE (extracted text)", tofile="AFTER (extracted text)", lineterm="",
        ))
        if len(comparison) > 20000:
            warnings.append("The change preview is shortened; inspect the stored source excerpts for context.")
        comparison = comparison[:20000]
    evidence = {
        "mode": "before_after" if before else "completed_work",
        "documents": documents,
        "comparison": comparison,
        "employee_notes": notes.strip(),
        "employee_confirmed": {key: value for key, value in (context or {}).items() if value.strip()},
    }
    draft.capture_evidence = evidence
    draft.source_file = "; ".join(dict.fromkeys(file.name for file in before + after + supporting)) or "employee explanation"
    # Keep evidence from each role represented in the bounded model context.
    chunks = [f"{doc['role'].upper()} | {doc['source_id']} | {doc['filename']}\n{str(doc['text'])[:3000]}" for doc in documents]
    draft.evidence_preview = "\n\n".join([
        f"EMPLOYEE NOTES:\n{notes[:4000]}",
        f"EMPLOYEE CONTEXT:\n{str(context or {})[:4000]}",
        f"OBSERVED TEXT CHANGES (not proof of motive or success):\n{comparison[:6000]}",
        *chunks,
    ])
    draft.readable_files = [str(doc["filename"]) for doc in documents if str(doc["text"]).strip()]
    draft.warnings = list(dict.fromkeys(draft.warnings + warnings))
    if before:
        draft.warnings.append("Text comparison identifies edits. Confirm why they were made and their result in the interview.")
    return draft
