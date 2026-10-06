"""Keep literal company evidence separate from generated recommendations."""


def evidence_only_recommendation(result) -> str:
    sections = [
        "## 1. Company-data-only recommendation",
        "The recorded approaches below are quoted from retrieved decisions, without added AI advice. "
        "They describe past situations; they are not automatically applicable to this one.",
    ]
    cases = list(result.similar_cases)[:3]
    if not cases:
        sections.append("No company decision was retrieved. There is no data-only recommendation available.")
    else:
        if result.knowledge_match < 0.35:
            sections.append("**Weak match:** these are the closest records, not sufficient evidence for a company-backed solution to your current problem.")
        for number, case in enumerate(cases, 1):
            sections.append(f"### Historical option {number}")
            for label, field in (
                ("Source", "source_file"), ("Original situation", "summary"),
                ("Recorded approach", "chosen_approach"), ("Recorded steps", "instructions"),
                ("Recorded reasoning", "reasoning"), ("Constraints", "constraints"),
                ("Exceptions", "exceptions"), ("Recorded outcome", "outcome"),
            ):
                value = str(getattr(case, field, "") or "").strip()
                if value:
                    sections.append(f"**{label}:**\n\n" + "\n".join("> " + line for line in value.splitlines()))
            if not getattr(case, "instructions", "") and not getattr(case, "chosen_approach", ""):
                sections.append("This record contains no explicit approach or steps; none have been invented.")
    return "\n\n".join(sections)


def two_recommendations(result, adapted: str, *, fallback: bool = False) -> str:
    if result.knowledge_match < 0.15:
        label = "No applicable company evidence was found for this request. " + ("This is rules-based general guidance." if fallback else "This is local AI general guidance.")
    else:
        label = "Rule-based adaptation; the local generative model is unavailable." if fallback else "AI adaptation of company evidence plus general reasoning."
    return (evidence_only_recommendation(result) + "\n\n---\n\n"
            "## 2. Adapted recommendation: company evidence + additional guidance\n\n"
            + label + " Added advice is not established company policy.\n\n" + adapted)
