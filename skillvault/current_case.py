"""Bounded, opt-in context for a worker's current question (not company knowledge)."""

from __future__ import annotations

from dataclasses import dataclass
import re

from .importer import WorkFile, extract_text


MAX_CONTEXT_BYTES = 8 * 1024 * 1024
MAX_CONTEXT_CHARS = 5000


@dataclass(frozen=True)
class CurrentCase:
    question: str
    team: str = ""
    filename: str = ""
    excerpt: str = ""
    warning: str = ""

    def model_question(self) -> str:
        parts = [self.question.strip() or "Help me with the attached current work."]
        if self.team:
            parts.append(f"Employee-provided team: {self.team}")
        if self.excerpt:
            parts.append(
                f"Current-case attachment ({self.filename}; employee-provided, not approved company knowledge):\n"
                f"{self.excerpt}"
            )
        return "\n\n".join(parts)

    def retrieval_question(self) -> str:
        # Keep the employee's question dominant; attached files are only a hint.
        return self.question.strip() + (f"\nCurrent work excerpt: {self.excerpt[:1200]}" if self.excerpt else "")


def prepare_current_case(question: str, team: str = "", file: WorkFile | None = None) -> CurrentCase:
    team = re.sub(r"\s+", " ", team).strip()[:120]
    if file is None:
        return CurrentCase(question.strip(), team)
    if len(file.data) > MAX_CONTEXT_BYTES:
        return CurrentCase(question.strip(), team, warning="Attachment is over 8 MB; it was not included. Summarize the relevant part or upload a smaller file.")
    extracted, warning = extract_text(file)
    if not extracted.strip():
        return CurrentCase(question.strip(), team, warning=warning or "No readable text was found in the attachment; describe the relevant part in your question.")
    return CurrentCase(question.strip(), team, file.name, extracted[:MAX_CONTEXT_CHARS], warning or "")


def material_followup(case: CurrentCase, match: float) -> str | None:
    """Ask at most one decision-changing question, after offering an initial plan."""
    text = (case.question + " " + case.excerpt).lower()
    if re.match(r"\s*(what is|define|explain|summarize)\b", case.question.lower()):
        return None
    if any(word in text for word in ("presentation", "slide", "deck", "pitch")):
        if not any(word in text for word in ("investor", "client", "customer", "executive", "board", "engineer", "audience")):
            return "Who is the audience, and what decision do they need to make after the presentation?"
        if match < 0.35 and not any(word in text for word in ("minute", "hour", "deadline", "time limit")):
            return "How much time do you have to present?"
    elif any(word in text for word in ("error", "bug", "crash", "fails", "failure", "api", "incident")):
        if not any(word in text for word in ("error code", "traceback", "expected", "actual", "logs show", "http ")):
            return "What exact error or unexpected result do you see, and what did you expect instead?"
    elif any(word in text for word in ("report", "dashboard", "metric", "kpi")):
        if not any(word in text for word in ("date range", "last week", "last month", "yesterday", "today", "q1", "q2", "q3", "q4")):
            return "What reporting period and metric definition should the answer use?"
    elif any(word in text for word in ("client", "customer", "tenant")) and match < 0.35:
        if not any(word in text for word in ("one user", "all users", "multiple", "everyone", "affected")):
            return "How many customers or users are affected, and is the issue still happening?"
    return None
