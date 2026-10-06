"""Persistent, bounded feedback adaptation; not language-model weight training."""

import re


RULES = {
    "specific": "Replace vague verbs with concrete actions, required inputs, and observable outputs. Name tools only when supported; otherwise say what capability is needed.",
    "steps": "Break the work into ordered, executable steps. Include preparation, the action, a success check, and what to do if it fails.",
    "detail": "Provide more explanation of how and why for each step, plus a worked example or usable draft. Do not add filler or invent details to increase length.",
    "concise": "Keep the answer compact, remove repetition, and preserve essential steps, source labels, and safety boundaries.",
    "clear": "Use plain language, explain technical terms, and keep each step focused on one action.",
    "recheck": "Previous similar advice was not helpful. Recheck whether each source applies, explicitly identify mismatches, and offer a different supported approach or ask the key missing question. Do not repeat a failed plan without addressing its limitation.",
}


def _tokens(text):
    stop = {"what", "with", "that", "this", "have", "help", "need", "would", "should", "please", "some", "does", "make", "about", "could", "from"}
    return set(re.findall(r"[a-z0-9]{4,}", str(text).lower())) - stop


def learning_profile(history, query, workspace_id):
    """Derive fixed guidance only; never promote arbitrary comments to instructions."""
    active = []
    for record in history:
        if record.get("workspace_id") != workspace_id:
            continue
        if record.get("event") == "reset_answer_learning":
            active = []
        elif record.get("learn_for_future", False):
            active.append(record)
    # Latest submission for the same reviewer/question replaces repeated clicks.
    unique = {}
    for record in active[-100:]:
        reviewer = record.get("reviewer", {})
        key = (str(reviewer.get("user_id", "")), str(record.get("query", "")).lower().strip())
        unique.pop(key, None)
        unique[key] = record
    enabled = set()
    length_rule = None
    matches = 0
    query_tokens = _tokens(query)
    for record in unique.values():
        issues = set(record.get("issue_types", []))
        ratings = record.get("ratings", {})
        if issues & {"Too generic", "Too vague", "Not specific enough"}:
            enabled.add("specific")
        if "Missing important steps" in issues or ratings.get("actionability", 3) <= 2:
            enabled.add("steps")
        if "Unclear explanation" in issues or ratings.get("clarity", 3) <= 2:
            enabled.add("clear")
        if "Too short" in issues:
            length_rule = "detail"
        if "Too long" in issues:
            length_rule = "concise"
        old_tokens = _tokens(record.get("query", ""))
        shared = query_tokens & old_tokens
        related = len(shared) >= 2 and len(shared) / max(1, min(len(query_tokens), len(old_tokens))) >= 0.3
        if related and (record.get("verdict") in {"Not helpful", "Needs improvement", "Incorrect or unsafe"} or "Not helpful" in issues):
            enabled.add("recheck")
            matches += 1
    if length_rule:
        enabled.add(length_rule)
    return {"rules": [key for key in RULES if key in enabled], "feedback_count": len(unique), "related_negative_count": matches}


def learning_instructions(profile):
    return [RULES[key] for key in (profile or {}).get("rules", []) if key in RULES]
