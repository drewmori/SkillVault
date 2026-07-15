from __future__ import annotations

from dataclasses import dataclass
import re
from collections import Counter

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

from .data import ExpertCase


@dataclass
class SimilarCase:
    summary: str
    label: str
    similarity: float
    reasoning: str
    source_file: str
    instructions: str
    software: str
    methods: str
    outcome: str
    media: list[dict[str, object]]


@dataclass
class Prediction:
    label: str
    confidence: float
    explanation: str
    similar_cases: list[SimilarCase]
    guidance: str
    sources: list[str]
    knowledge_match: float
    general_suggestions: list[str]


@dataclass
class LearningResult:
    score: float
    matched_concepts: list[str]
    missing_concepts: list[str]
    feedback: str


class SkillVaultEngine:
    def __init__(self, cases: list[ExpertCase]) -> None:
        self.cases = cases
        self.labels = sorted({case.label for case in cases})
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.classifier = LogisticRegression(max_iter=1000, random_state=42)
        self.matrix = None
        self.validation_accuracy = 0.0

    def train(self) -> None:
        texts = [self._case_text(case) for case in self.cases]
        labels = [case.label for case in self.cases]
        self.matrix = self.vectorizer.fit_transform(texts)

        # The dataset is intentionally small for this prototype, so report a
        # holdout score only when every class can appear in both splits.
        if len(self.cases) >= 12:
            label_counts = Counter(labels)
            # Stratified splitting requires at least two examples per class.
            # Small, realistic company datasets may not satisfy that yet.
            stratify_labels = labels if min(label_counts.values()) >= 2 else None
            train_texts, test_texts, train_labels, test_labels = train_test_split(
                texts, labels, test_size=0.25, random_state=42, stratify=stratify_labels
            )
            evaluator = LogisticRegression(max_iter=1000, random_state=42)
            evaluator.fit(self.vectorizer.transform(train_texts), train_labels)
            self.validation_accuracy = accuracy_score(test_labels, evaluator.predict(self.vectorizer.transform(test_texts)))

        self.classifier.fit(self.matrix, labels)

    def replace_cases(self, cases: list[ExpertCase]) -> None:
        if len(cases) < 3:
            raise ValueError("At least three approved cases are needed to train SkillVault.")
        self.cases = cases
        self.labels = sorted({case.label for case in cases})
        self.train()

    def predict(self, text: str, scenario: str = "All scenarios") -> Prediction:
        vector = self.vectorizer.transform([text])
        probabilities = self.classifier.predict_proba(vector)[0]
        index = probabilities.argmax()
        label = str(self.classifier.classes_[index])
        confidence = float(probabilities[index])
        similar = self._similar_cases(vector, text, scenario)
        explanation = self._explanation(label, confidence, similar)
        guidance = self._guidance(similar)
        sources = list(dict.fromkeys(case.source_file for case in similar))
        knowledge_match = similar[0].similarity if similar else 0.0
        return Prediction(label, confidence, explanation, similar, guidance, sources, knowledge_match, self._general_suggestions(text) if confidence < 0.75 or knowledge_match < 0.35 else [])

    def _similar_cases(self, vector, text: str = "", scenario: str = "All scenarios", limit: int = 3) -> list[SimilarCase]:
        """Retrieve cases with semantic similarity plus explicit intent routing."""
        scores = cosine_similarity(vector, self.matrix)[0]
        ranking_scores = scores.copy()
        lowered = text.lower()

        allowed = [index for index, case in enumerate(self.cases) if self._matches_scenario(case, scenario)]
        if allowed and scenario != "All scenarios":
            ranking_scores[:] = -1.0
            ranking_scores[allowed] = scores[allowed]

        # TF-IDF alone can rank unrelated short queries almost randomly. Give
        # the user's clear intent priority while preserving the raw similarity
        # score shown in the UI.
        intent_weights = [
            (("presentation", "powerpoint", "investor", "slide", "pitch", "deck"), "presentation", 0.65),
            (("report", "dashboard", "metric", "csv", "kpi", "retention"), "report", 0.45),
            (("python", "code", "api", "function", "error", "bug", "crash"), "code", 0.4),
            (("security", "privacy", "secret", "delete", "production access"), "urgent", 0.35),
            (("client", "customer", "tenant", "user"), "client_issue", 0.2),
        ]
        for keywords, feature, weight in intent_weights:
            if any(keyword in lowered for keyword in keywords):
                for index, case in enumerate(self.cases):
                    if case.features.get(feature, 0):
                        ranking_scores[index] += weight
                    elif feature == "presentation" and case.features.get("presentation", 0) == 0:
                        ranking_scores[index] -= 0.08

        ranked = ranking_scores.argsort()[::-1][:limit]
        return [SimilarCase(self.cases[i].summary, self.cases[i].label, float(scores[i]), self.cases[i].reasoning, self.cases[i].source_file, self.cases[i].instructions, self.cases[i].software, self.cases[i].methods, self.cases[i].outcome, self.cases[i].media) for i in ranked]

    def _matches_scenario(self, case: ExpertCase, scenario: str) -> bool:
        if scenario == "All scenarios":
            return True
        lowered = " ".join((case.summary, case.reasoning, case.software, case.methods)).lower()
        if scenario == "Client support":
            return bool(case.features.get("client_issue") or any(word in lowered for word in ("client", "customer", "support", "tenant")))
        if scenario == "Engineering/code":
            return bool(case.features.get("code") or any(word in lowered for word in ("api", "python", "code", "test", "deployment", "webhook")))
        if scenario == "Presentations & pitches":
            return bool(case.features.get("presentation") or any(word in lowered for word in ("presentation", "powerpoint", "investor", "pitch", "slide", "deck")))
        if scenario == "Reporting & analytics":
            return bool(case.features.get("report") or any(word in lowered for word in ("report", "metric", "dashboard", "analytics", "csv")))
        if scenario == "Security & access":
            return any(word in lowered for word in ("security", "sso", "saml", "scim", "oauth", "permission", "access", "certificate", "secret"))
        if scenario == "Billing & operations":
            return any(word in lowered for word in ("billing", "invoice", "payment", "revenue", "backup", "restore", "incident", "outage", "alert"))
        if scenario == "Onboarding & documentation":
            return any(word in lowered for word in ("onboarding", "documentation", "runbook", "new hire", "meeting", "workflow"))
        if scenario == "Other":
            known = ("client", "customer", "support", "api", "python", "code", "test", "presentation", "powerpoint", "investor", "report", "metric", "dashboard", "security", "sso", "permission", "billing", "incident", "onboarding", "documentation")
            return not any(word in lowered for word in known)
        return True

    def _case_text(self, case: ExpertCase) -> str:
        return " ".join((case.summary, case.reasoning, case.instructions, case.software, case.methods, case.outcome))

    def _explanation(self, label: str, confidence: float, similar: list[SimilarCase]) -> str:
        if confidence < 0.75:
            tone = "The model is uncertain because this case does not closely match the expert examples."
        else:
            tone = "The recommendation is supported by patterns in the expert examples."
        examples = ", ".join(case.label.lower() for case in similar[:2])
        return f"{tone} The closest historical cases led to: {examples}. The recommended approach is {label.lower()}, with the detailed playbook below."

    def _guidance(self, similar: list[SimilarCase]) -> str:
        lines = ["1. Start with the closest expert example and identify which facts match the new case."]
        for index, case in enumerate(similar[:3], start=2):
            instruction = case.instructions or case.reasoning
            lines.append(f"{index}. {instruction} Use: {case.software}. Method: {case.methods}. (Source: `{case.source_file}`)")
        lines.append(f"{len(lines) + 1}. If the new case has missing or conflicting information, pause and request human review.")
        return "\n\n".join(lines)

    def _general_suggestions(self, text: str) -> list[str]:
        """Offer clearly-labeled baseline ideas when expert evidence is weak."""
        lowered = text.lower()
        if any(word in lowered for word in ("error", "bug", "crash", "broken", "debug", "fix", "fails")):
            return [
                "Reproduce the issue with the smallest input that still fails.",
                "Read the complete error message and traceback before changing code.",
                "Check what changed most recently and compare against the last working version.",
                "Create a regression test so the same failure is caught in the future.",
            ]
        if any(word in lowered for word in ("presentation", "slide", "deck", "talk")):
            return [
                "Define the audience and the one decision or idea the presentation should communicate.",
                "Give each slide one main message and use visuals to support it.",
                "Move detailed evidence into speaker notes or an appendix.",
                "Rehearse once while timing the talk and mark places where the story is unclear.",
            ]
        if any(word in lowered for word in ("make", "build", "create", "design")):
            return [
                "Write down the desired result and the constraints before choosing tools.",
                "Build the smallest useful version first.",
                "Test it with a real example and record what went wrong.",
                "Improve one failure at a time instead of adding every feature at once.",
            ]
        if any(word in lowered for word in ("code", "python", "javascript", "function", "program")):
            return [
                "Define the input, output, and edge cases before implementing.",
                "Start with a small example that proves the core behavior.",
                "Add tests for both the normal path and likely failure cases.",
                "Keep the change small enough that you can explain every line of it.",
            ]
        return [
            "Clarify the desired outcome and what constraints matter most.",
            "Break the task into the smallest verifiable steps.",
            "Try a low-risk first version and inspect the result before expanding it.",
            "Ask for human review if the decision affects safety, money, privacy, or other people.",
        ]

    def _adaptation_notes(self, query: str, result: Prediction) -> str:
        """Explain the concrete gap between the retrieved case and today's request."""
        lowered = query.lower()
        notes = []
        closest = result.similar_cases[0] if result.similar_cases else None
        quantities = re.findall(r"\b\d+(?:\.\d+)?\s*(?:mb|gb|kb|%|users?|customers?|hours?|days?|seconds?)?\b", query, flags=re.IGNORECASE)
        if quantities:
            notes.append(f"Today's request includes a measurable boundary or scope ({', '.join(quantities)}), so verify that boundary directly instead of copying the old case's assumptions.")
        if any(word in lowered for word in ("only", "while", "but", "although", "except")):
            notes.append("Today's request has a condition where the issue works in one situation but fails in another; compare the working and failing paths side by side.")
        if any(word in lowered for word in ("after", "since", "following", "immediately", "recently")):
            notes.append("The timing in today's request matters, so compare what changed immediately before the issue started with the last known-good state.")
        if any(word in lowered for word in ("urgent", "outage", "blocked", "many customers", "production")):
            notes.append("The current impact is higher-risk than a routine case, so preserve evidence and involve the owner before applying a broad change.")
        if closest:
            notes.append(f"The historical case was about: {closest.summary} Today's plan keeps its useful method but tests whether that method actually explains this new request.")
        if not notes:
            notes.append("The current request is not identical to the retrieved case, so validate the matching facts first and drop any historical step that does not apply.")
        return "\n\n".join(f"- {note}" for note in notes)

    def _tailored_steps(self, query: str, result: Prediction) -> str:
        """Adapt retrieved directions to the facts and risk of the new request."""
        lowered = query.lower()
        closest = result.similar_cases[0] if result.similar_cases else None
        historical_lines = []
        if closest:
            historical_lines = [
                line.strip().lstrip("0123456789.- ")
                for line in (closest.instructions or closest.reasoning).splitlines()
                if line.strip()
            ]

        if any(word in lowered for word in ("code", "python", "api", "error", "bug", "crash", "fails", "failure")):
            opening = [
                "1. Translate today's report into a smallest reproducible example: capture the exact input, expected result, actual result, timestamp, and environment.",
                "2. Run the reproduction before changing code and save the complete error, traceback, request, or response with secrets removed.",
                "3. Compare the failing path with the last known-good version or request, then isolate one likely cause at a time.",
            ]
            closing = [
                "4. Apply the narrowest reversible change and add a regression test for this exact failure.",
                "5. Verify normal, missing, null, boundary, and retry inputs before opening a handoff or release.",
            ]
        elif any(word in lowered for word in ("report", "dashboard", "metric", "csv", "kpi", "retention")):
            opening = [
                "1. Freeze the exact report question, date range, timezone, filters, account scope, and source-system version from today's request.",
                "2. Reconcile one small sample manually before changing the dashboard or publishing a conclusion.",
                "3. Separate confirmed numbers from possible explanations and record the metric definition used.",
            ]
            closing = [
                "4. Show the result in the audience's language with one clear takeaway and a visible source/date note.",
                "5. Ask the data owner to review any conflicting definition or unexplained discrepancy.",
            ]
        elif any(word in lowered for word in ("presentation", "slide", "deck", "pitch")):
            opening = [
                "1. Identify today's audience, decision, and one message the presentation must communicate.",
                "2. Rebuild the story around the audience's problem and the smallest convincing example, rather than carrying over every old slide.",
                "3. Put one conclusion in each slide title and move supporting detail to notes or an appendix.",
            ]
            closing = [
                "4. Check every claim against the available evidence and label assumptions clearly.",
                "5. End with the exact decision, ask, owner, and next date, then rehearse once with a timer.",
            ]
        elif any(word in lowered for word in ("security", "secret", "key", "privacy", "delete", "production access")):
            opening = [
                "1. Stop any irreversible or privilege-expanding action and identify the tenant, system, scope, and time involved.",
                "2. Minimize sensitive data in notes and attachments; do not copy secrets or customer records into a broader ticket.",
                "3. Check the approved security, privacy, retention, or production-access process before investigating further.",
            ]
            closing = [
                "4. Preserve only the evidence the responsible owner needs and record what is still unknown.",
                "5. Escalate to the designated security, privacy, or data owner before making a final decision.",
            ]
        elif any(word in lowered for word in ("client", "customer", "user", "tenant")):
            opening = [
                "1. Confirm the affected tenant, users, exact impact, timeline, and what the client already tried.",
                "2. Match today's facts against the closest historical case instead of assuming the old resolution applies unchanged.",
                "3. Reproduce or verify the issue with a low-risk test and keep the client updated on what is confirmed versus unknown.",
            ]
            closing = [
                "4. Give the client a concrete next action, owner, and update time; do not promise a permanent fix before verification.",
                "5. Escalate with a concise evidence package if the issue repeats, affects multiple customers, or needs an exception.",
            ]
        else:
            opening = [
                "1. Define the outcome, constraints, people affected, and evidence available for today's request.",
                "2. Use the closest historical decision as a starting point, then remove steps that do not apply to the current facts.",
                "3. Try the smallest reversible version and record what happened.",
            ]
            closing = [
                "4. Verify the result against a specific success condition and document the decision.",
                "5. Request human review when the evidence is weak or the decision affects security, money, privacy, or production.",
            ]

        adapted = opening + closing
        if historical_lines:
            adapted.insert(3, f"Historical judgment carried forward: {historical_lines[0]}")
        return "\n\n".join(adapted)

    def compose_response(self, query: str, result: Prediction) -> str:
        """Write a complete local answer from retrieved expert decisions."""
        evidence = result.similar_cases
        tools = list(dict.fromkeys(case.software for case in evidence if case.software))
        methods = list(dict.fromkeys(case.methods for case in evidence if case.methods))
        sources = list(dict.fromkeys(case.source_file for case in evidence if case.source_file))
        weak_match = result.knowledge_match < 0.35 or result.confidence < 0.75
        escalation = (
            "Escalate for human review before making a final change. The closest examples are not strong enough to support a confident company-specific answer."
            if weak_match else
            "Proceed with the playbook, then ask for review if the facts differ from the retrieved examples or if the issue affects security, privacy, billing, or production availability."
        )
        tool_text = ", ".join(tools) if tools else "No specific software was recorded in the retrieved cases."
        method_text = "; ".join(methods) if methods else "Use the numbered verification steps in the playbook."
        source_text = "\n".join(f"- `{source}`" for source in sources) or "- No source file was recorded."
        general = "\n".join(f"- {item}" for item in result.general_suggestions) or "- No extra general suggestions were needed because the expert evidence was a close match."
        lowered = query.lower()
        if any(word in lowered for word in ("presentation", "powerpoint", "investor", "slide", "pitch", "deck")):
            draft_title = "#### Draft investor presentation structure"
            draft_body = (
                "1. **Problem:** Who has the problem and why it matters now.\n"
                "2. **Solution:** Show the smallest product workflow that solves it.\n"
                "3. **Evidence:** Add verified traction, customer, usage, or outcome numbers with dates and sources.\n"
                "4. **Business:** Explain the market, business model, differentiation, and competition.\n"
                "5. **Ask:** State the funding request, use of funds, milestones, and next step.\n\n"
                "Move technical architecture, detailed metrics, and extra features to an appendix."
            )
        elif any(word in lowered for word in ("report", "dashboard", "metric", "csv", "kpi")):
            draft_title = "#### Draft report structure"
            draft_body = (
                "1. **Takeaway:** State the decision or finding in one sentence.\n"
                "2. **Definition:** Name the metric, date range, filters, and timezone.\n"
                "3. **Evidence:** Show the verified numbers and source file.\n"
                "4. **Interpretation:** Separate confirmed causes from hypotheses.\n"
                "5. **Action:** End with an owner, deadline, and next decision."
            )
        elif any(word in lowered for word in ("code", "python", "api", "error", "bug", "crash", "fails")):
            draft_title = "#### Draft engineering handoff"
            draft_body = (
                f"**Issue:** {query.strip()}\n\n"
                f"**Recommended path:** {result.label}.\n\n"
                "**Include:** exact reproduction, expected versus actual behavior, environment/version, timestamps, sanitized logs, recent changes, customer impact, and remaining unknowns."
            )
        else:
            draft_title = "#### Draft response or next-step note"
            draft_body = (
                f"**Request:** {query.strip()}\n\n"
                f"**Recommended path:** {result.label}.\n\n"
                "State what is confirmed, what is still unknown, the next action, the owner, and the next update time."
            )
        tailored_steps = self._tailored_steps(query, result)
        adaptation_notes = self._adaptation_notes(query, result)
        historical_anchor = (
            result.similar_cases[0].instructions or result.similar_cases[0].reasoning
            if result.similar_cases else
            "No historical direction was available."
        )
        return f"""**Recommended approach: {result.label}**

{result.explanation}

#### What changed for today's request

{adaptation_notes}

#### Tailored directions for this request

These are the actions SkillVault changed or added for the issue entered today:

{tailored_steps}

#### Historical judgment carried forward

{historical_anchor}

#### Tools and methods from the expert cases

- **Software/tools:** {tool_text}
- **Methods:** {method_text}

#### Escalation decision

{escalation}

{draft_title}

{draft_body}

#### General suggestions beyond the company knowledge

{general}

#### Evidence used

{source_text}
"""

    def teaching_feedback(self, case: ExpertCase) -> str:
        if case.label == "Build in small steps":
            return "The expert decomposes larger tasks into a small first version, tests it with a real example, and adds complexity only after the basics work."
        if case.label == "Write and test":
            return "The expert protects against regressions by defining expected behavior and testing a small change before expanding it."
        if case.label == "Diagnose and verify":
            return "The expert gathers evidence and reproduces the problem before changing anything, which prevents guessing and makes the fix repeatable."
        if case.label == "Escalate for review":
            return "The expert recognizes uncertainty and routes the case for human judgment instead of presenting a confident unsupported answer."
        if case.label == "Follow standard process":
            return "The expert recognizes a familiar pattern and applies the established workflow consistently."
        if case.features["late_request"] and case.features["defect"]:
            return "Notice the tension between timing and evidence: a late request with a verified defect is usually escalated rather than automatically approved."
        if case.features["defect"] and case.features["evidence"]:
            return "Verified evidence of a product defect is one of the strongest signals in the expert's decisions."
        if case.features["late_request"]:
            return "Late requests without credible defect evidence were usually denied by the former expert."
        return "Timely requests generally received a simpler resolution unless the item was high value or the situation was ambiguous."

    def evaluate_learning(self, case: ExpertCase, learner_answer: str) -> LearningResult:
        """Give transparent practice feedback using the expert playbook."""
        target = case.instructions or case.reasoning
        stop_words = {"the", "and", "then", "with", "from", "that", "this", "into", "before", "after", "for", "use", "one"}
        target_words = [word for word in re.findall(r"[a-zA-Z]{4,}", target.lower()) if word not in stop_words]
        answer_words = set(word for word in re.findall(r"[a-zA-Z]{4,}", learner_answer.lower()) if word not in stop_words)
        concepts = list(dict.fromkeys(target_words))
        matched = [word for word in concepts if word in answer_words]
        missing = [word for word in concepts if word not in answer_words][:6]
        score = min(1.0, len(matched) / max(1, min(8, len(concepts))))
        if score >= 0.7:
            feedback = "Strong answer. You covered the main ideas from the expert playbook."
        elif score >= 0.4:
            feedback = "Good start. Your answer overlaps with the expert approach, but it is missing a few important steps."
        else:
            feedback = "Your answer is a useful starting point, but it does not yet capture enough of the expert's process. Compare it with the playbook and try again."
        return LearningResult(score, matched[:8], missing, feedback)

    def add_feedback(self, summary: str, label: str) -> None:
        self.cases.append(ExpertCase(summary, label, "Added through human expert feedback.", {"age_days": 0, "defect": 0, "evidence": 0, "repeat_customer": 0, "late_request": 0, "high_value": 0}))
        self.train()

    def add_expert_case(self, case: ExpertCase) -> None:
        self.add_expert_cases([case])

    def add_expert_cases(self, cases: list[ExpertCase]) -> None:
        self.cases.extend(cases)
        self.labels = sorted({case.label for case in self.cases})
        self.train()

    def case_table(self) -> pd.DataFrame:
        return pd.DataFrame([{"Case": i + 1, "Expert approach": case.label, "Scenario": case.summary, "Software/tools": case.software, "Methods": case.methods, "Source file": case.source_file, "Outcome": case.outcome, "Reasoning": case.reasoning} for i, case in enumerate(self.cases)])
