from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from collections import Counter

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

from .data import ExpertCase
from .search_index import PersistentKnowledgeIndex
from .semantic_search import SemanticIndex, SemanticSearchError


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
    expert_owner: str
    department: str
    approval_date: str
    last_reviewed_date: str
    expiration_date: str
    lifecycle_status: str
    replaces_source: str
    goal: str
    chosen_approach: str
    alternatives_considered: str
    constraints: str
    reusable_rule: str
    exceptions: str
    capture_evidence: dict[str, object] = field(default_factory=dict)


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
    retrieval_mode: str = "In-memory TF-IDF vectors"
    candidate_count: int = 0
    indexed_count: int = 0


@dataclass
class LearningResult:
    score: float
    matched_concepts: list[str]
    missing_concepts: list[str]
    feedback: str


class SkillVaultEngine:
    def __init__(self, cases: list[ExpertCase], index_path: Path | None = None, semantic_index: SemanticIndex | None = None) -> None:
        self.cases = cases
        self.labels = sorted({case.label for case in cases})
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.classifier = LogisticRegression(max_iter=1000, random_state=42)
        self.matrix = None
        self.validation_accuracy = 0.0
        self.is_trained = False
        self.uses_classifier = False
        self.search_index = PersistentKnowledgeIndex(index_path) if index_path is not None else None
        self.index_backend = "In-memory TF-IDF vectors"
        self.index_rebuilt = False
        self.last_candidate_count = 0
        self.semantic_index = semantic_index
        self.semantic_indices = []
        self.semantic_matrix = None
        self.semantic_status = "Keyword search"

    def train(self) -> None:
        self.labels = sorted({case.label for case in self.cases})
        self.validation_accuracy = 0.0
        self.is_trained = False
        self.uses_classifier = False
        if not self.cases:
            self.matrix = None
            return

        texts = [self._case_text(case) for case in self.cases]
        labels = [case.label for case in self.cases]
        if self.search_index is not None:
            self.vectorizer, self.matrix, index_stats = self.search_index.sync(self.cases)
            self.index_backend = index_stats.backend
            self.index_rebuilt = index_stats.rebuilt
        else:
            self.matrix = self.vectorizer.fit_transform(texts)
            self.index_backend = "In-memory TF-IDF vectors"
            self.index_rebuilt = False
        self.is_trained = True
        if self.semantic_index:
            self.semantic_indices, self.semantic_matrix = self.semantic_index.load(self.cases)
            self.semantic_status = f"{len(self.semantic_indices)} of {len(self.cases)} decisions embedded"

        # The dataset is intentionally small for this prototype, so report a
        # holdout score only when every class can appear in both splits.
        if len(self.cases) >= 12 and len(self.labels) >= 2:
            label_counts = Counter(labels)
            # Skip the holdout score until every class can safely appear in
            # both splits. Retrieval and full-data training still work while
            # a clean workspace is building its first few decision classes.
            test_count = max(1, round(len(self.cases) * 0.25))
            if min(label_counts.values()) >= 2 and test_count >= len(self.labels):
                train_texts, test_texts, train_labels, test_labels = train_test_split(
                    texts, labels, test_size=0.25, random_state=42, stratify=labels
                )
                evaluator = LogisticRegression(max_iter=1000, random_state=42)
                evaluator.fit(self.vectorizer.transform(train_texts), train_labels)
                self.validation_accuracy = accuracy_score(test_labels, evaluator.predict(self.vectorizer.transform(test_texts)))

        if len(self.labels) >= 2:
            self.classifier.fit(self.matrix, labels)
            self.uses_classifier = True

    def replace_cases(self, cases: list[ExpertCase]) -> None:
        if len(cases) < 3:
            raise ValueError("At least three approved cases are needed to train SkillVault.")
        self.cases = cases
        self.labels = sorted({case.label for case in cases})
        self.train()

    def predict(self, text: str, scenario: str = "All scenarios") -> Prediction:
        if not self.is_trained or self.matrix is None:
            raise ValueError("SkillVault needs at least one approved expert decision before it can answer.")
        vector = self.vectorizer.transform([text])
        similar = self._similar_cases(vector, text, scenario)
        knowledge_match = similar[0].similarity if similar else 0.0
        if self.uses_classifier:
            probabilities = self.classifier.predict_proba(vector)[0]
            index = probabilities.argmax()
            label = str(self.classifier.classes_[index])
            classifier_confidence = float(probabilities[index])
        else:
            # A brand-new workspace may begin with one approved decision. In
            # that case retrieval still works, but confidence stays evidence-
            # based until multiple decision classes are available.
            label = self.labels[0]
            classifier_confidence = 0.50 + (0.30 * knowledge_match)
        # The UI confidence represents the quality of the recommendation's
        # support, not only the classifier's probability across five labels.
        # A strong retrieved expert match should raise confidence, while a
        # weak match should continue to trigger clarification and review.
        evidence_confidence = 0.50 + (0.50 * knowledge_match)
        confidence = min(0.98, max(classifier_confidence, evidence_confidence))
        explanation = self._explanation(label, confidence, similar)
        guidance = self._guidance(similar)
        sources = list(dict.fromkeys(case.source_file for case in similar))
        return Prediction(
            label,
            confidence,
            explanation,
            similar,
            guidance,
            sources,
            knowledge_match,
            self._general_suggestions(text) if confidence < 0.75 or knowledge_match < 0.35 else [],
            retrieval_mode=self.index_backend,
            candidate_count=self.last_candidate_count,
            indexed_count=len(self.cases),
        )

    def _similar_cases(self, vector, text: str = "", scenario: str = "All scenarios", limit: int = 5) -> list[SimilarCase]:
        """Retrieve cases with semantic similarity plus explicit intent routing."""
        lowered = text.lower()

        allowed = [index for index, case in enumerate(self.cases) if self._matches_scenario(case, scenario)]
        if not allowed:
            allowed = list(range(len(self.cases)))
        allowed_filter = allowed if scenario != "All scenarios" else None
        if self.search_index is not None:
            candidates = self.search_index.candidate_indices(text, allowed_filter, limit=max(256, limit * 40))
        else:
            candidates = allowed if allowed_filter is not None else list(range(len(self.cases)))
        if not candidates:
            candidates = allowed

        semantic_scores = {}
        if self.semantic_index and self.semantic_matrix is not None:
            try:
                semantic_scores = self.semantic_index.search(text, self.semantic_indices, self.semantic_matrix, set(allowed))
                top_semantic = sorted(semantic_scores, key=semantic_scores.get, reverse=True)[:256]
                candidates = list(dict.fromkeys([*candidates, *top_semantic]))
            except SemanticSearchError:
                semantic_scores = {}
        self.index_backend = (
            "Local semantic embeddings + keyword search" if semantic_scores
            else "Keyword search (SQLite/TF-IDF)"
        )

        scores = cosine_similarity(vector, self.matrix[candidates])[0]
        if semantic_scores:
            scores = scores.copy()
            for position, index in enumerate(candidates):
                if index in semantic_scores:
                    scores[position] = 0.35 * scores[position] + 0.65 * max(0.0, semantic_scores[index])
        ranking_scores = scores.copy()
        self.last_candidate_count = len(candidates)

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
            weight *= 0.15 if semantic_scores else 1.0
            if any(keyword in lowered for keyword in keywords):
                for candidate_position, index in enumerate(candidates):
                    case = self.cases[index]
                    if case.features.get(feature, 0):
                        ranking_scores[candidate_position] += weight
                    elif feature == "presentation" and case.features.get("presentation", 0) == 0:
                        ranking_scores[candidate_position] -= 0.08

        ranked_positions = ranking_scores.argsort()[::-1][:limit]
        return [
            SimilarCase(
                self.cases[candidates[position]].summary,
                self.cases[candidates[position]].label,
                float(scores[position]),
                self.cases[candidates[position]].reasoning,
                self.cases[candidates[position]].source_file,
                self.cases[candidates[position]].instructions,
                self.cases[candidates[position]].software,
                self.cases[candidates[position]].methods,
                self.cases[candidates[position]].outcome,
                self.cases[candidates[position]].media,
                self.cases[candidates[position]].expert_owner,
                self.cases[candidates[position]].department,
                self.cases[candidates[position]].approval_date,
                self.cases[candidates[position]].last_reviewed_date,
                self.cases[candidates[position]].expiration_date,
                self.cases[candidates[position]].lifecycle_status,
                self.cases[candidates[position]].replaces_source,
                self.cases[candidates[position]].goal,
                self.cases[candidates[position]].chosen_approach,
                self.cases[candidates[position]].alternatives_considered,
                self.cases[candidates[position]].constraints,
                self.cases[candidates[position]].reusable_rule,
                self.cases[candidates[position]].exceptions,
                self.cases[candidates[position]].capture_evidence,
            )
            for position in ranked_positions
        ]

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
        return " ".join(
            (
                case.summary,
                case.reasoning,
                case.instructions,
                case.software,
                case.methods,
                case.outcome,
                case.expert_owner,
                case.department,
                case.goal,
                case.chosen_approach,
                case.alternatives_considered,
                case.constraints,
                case.reusable_rule,
                case.exceptions,
            )
        )

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

    @staticmethod
    def _contains_signal(text: str, signals: tuple[str, ...]) -> bool:
        """Match intent words without accidental substrings such as POS in reproduce."""
        for signal in signals:
            if signal.endswith("*"):
                prefix = re.escape(signal[:-1])
                if re.search(rf"\b{prefix}[a-zA-Z0-9_-]*\b", text):
                    return True
            elif re.search(rf"(?<![a-zA-Z0-9_]){re.escape(signal)}(?![a-zA-Z0-9_])", text):
                return True
        return False

    def _general_suggestions(self, text: str) -> list[str]:
        """Offer clearly-labeled baseline ideas when expert evidence is weak."""
        lowered = text.lower()
        if self._contains_signal(lowered, ("allergen*", "recall*", "freezer*", "temperature*", "contaminat*", "food safety", "unsafe product*")):
            return [
                "Stop sale or use of the affected product and physically isolate it while safety is unknown.",
                "Preserve the current temperature, label, lot, quantity, timing, and customer-exposure evidence.",
                "Do not taste, relabel, discard, reset equipment, or return product to sale before the responsible safety owner reviews it.",
                "Record the authorized disposition and verify containment before normal operation resumes.",
            ]
        if self._contains_signal(lowered, ("pos", "register*", "card payment*", "payment gateway", "authorization*", "offline mode", "duplicate charge*")):
            return [
                "Do not repeatedly retry the same card while authorization status is unknown.",
                "Compare all registers, store-network status, and the payment-provider status before selecting a workaround.",
                "Use offline processing only under the current approved incident procedure and transaction limits.",
                "Reconcile every queued authorization and receipt after recovery before closing the incident.",
            ]
        if self._contains_signal(lowered, ("return*", "refund*", "receipt*", "damaged item*", "exchange*")):
            return [
                "Verify the purchase, item, date, price, customer identity, and original tender independently.",
                "Document the item condition and obtain the required exception approval before refunding.",
                "Return funds only through an approved tender path and keep damaged merchandise out of saleable inventory.",
                "Record the reason, approver, refund result, and inventory or claims disposition.",
            ]
        if self._contains_signal(lowered, ("inventory", "markdown*", "clearance", "sell-through", "floor reset", "seasonal")):
            return [
                "Reconcile current inventory and open orders before changing price or transferring units.",
                "Compare sell-through, weeks of supply, margin, timing, and transfer cost by location.",
                "Prefer a measurable staged action when the deadline permits.",
                "Recheck units and margin before expanding the decision across more locations.",
            ]
        if self._contains_signal(lowered, ("staffing", "labor", "schedule*", "checkout line*", "traffic", "payroll", "shift*")):
            return [
                "Measure current traffic, wait time, conversion, coverage, and labor by time window.",
                "Move verified excess coverage into peak periods before requesting more payroll.",
                "Pilot the change in representative locations while preserving breaks and required coverage.",
                "Compare service and payroll outcomes before expanding the schedule.",
            ]
        if self._contains_signal(lowered, ("error*", "bug*", "crash*", "broken", "debug*", "fix*", "fail*")):
            return [
                "Reproduce the issue with the smallest input that still fails.",
                "Read the complete error message and traceback before changing code.",
                "Check what changed most recently and compare against the last working version.",
                "Create a regression test so the same failure is caught in the future.",
            ]
        if self._contains_signal(lowered, ("presentation*", "slide*", "deck*", "talk*")):
            return [
                "Define the audience and the one decision or idea the presentation should communicate.",
                "Give each slide one main message and use visuals to support it.",
                "Move detailed evidence into speaker notes or an appendix.",
                "Rehearse once while timing the talk and mark places where the story is unclear.",
            ]
        if self._contains_signal(lowered, ("make", "build*", "create*", "design*")):
            return [
                "Write down the desired result and the constraints before choosing tools.",
                "Build the smallest useful version first.",
                "Test it with a real example and record what went wrong.",
                "Improve one failure at a time instead of adding every feature at once.",
            ]
        if self._contains_signal(lowered, ("code", "python", "javascript", "function*", "program*")):
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

    def _instruction_lines(self, text: str) -> list[str]:
        """Split an expert playbook into reusable actions without list markers."""
        chunks = re.split(r"\n+|(?<=\.)\s+(?=\d+[.)]\s+)", text)
        actions = []
        for chunk in chunks:
            cleaned = re.sub(r"^(?:step\s+)?\d+[.)]\s*|^[-*•]\s*", "", chunk.strip(), flags=re.IGNORECASE)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            if len(cleaned) >= 18:
                actions.append(cleaned)
        if not actions:
            actions = [
                sentence.strip()
                for sentence in re.split(r"(?<=[.!?])\s+", text)
                if len(sentence.strip()) >= 18
            ]
        return actions[:10]

    def _important_query_terms(self, query: str, comparison: str = "") -> list[str]:
        stop_words = {
            "about", "after", "again", "also", "because", "before", "could", "does",
            "from", "have", "help", "into", "like", "need", "question", "should",
            "someone", "something", "that", "their", "there", "these", "they",
            "this", "what", "when", "where", "which", "with", "would",
        }
        comparison_words = set(re.findall(r"[a-zA-Z]{4,}", comparison.lower()))
        terms = []
        for word in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{3,}", query.lower()):
            if word not in stop_words and word not in comparison_words and word not in terms:
                terms.append(word)
        return terms[:7]

    def _answer_profile(self, query: str) -> tuple[str, list[str], list[str], list[tuple[tuple[str, ...], str]]]:
        """Return domain-specific opening steps, closing checks, and missing-context questions."""
        lowered = query.lower()
        if self._contains_signal(lowered, ("allergen*", "recall*", "freezer*", "temperature*", "contaminat*", "food safety", "unsafe product*")):
            return (
                "safety incident",
                [
                    "Protect customers first: stop the affected sale or activity, isolate the scope, and preserve the current evidence before changing equipment, labels, or records.",
                    "Record the current product or equipment identifier, first observed time, measured condition, affected locations, quantities, and any customer exposure.",
                ],
                [
                    "Use the designated safety or incident owner to decide disposition; do not infer that a product is safe because a past case was resolved.",
                    "Verify containment, inventory reconciliation, corrective action, and owner approval before returning anything to normal operation.",
                ],
                [
                    (("temperature", "allergen", "lot", "sku", "product"), "What exact product, equipment, lot/SKU, or measured condition is affected?"),
                    (("time", "since", "hour", "day", "today"), "When was the issue first observed, and what is the last verified normal time?"),
                    (("sold", "customer", "exposure", "used"), "Could any customer already have received or used the affected product?"),
                    (("food safety", "incident owner", "facilities", "safety team"), "Which safety, facilities, or incident owner has authority to approve the final disposition?"),
                ],
            )
        if self._contains_signal(lowered, ("pos", "register*", "card payment*", "payment gateway", "authorization*", "offline mode", "duplicate charge*")):
            return (
                "payment or POS incident",
                [
                    "Capture one current failure without repeatedly charging the same card, then determine whether the problem affects one register, every register, or the wider store network.",
                    "Compare register diagnostics, network reachability, provider status, and the first failure time before enabling any workaround.",
                ],
                [
                    "Use offline processing only if the current incident owner and company policy explicitly allow it, including current transaction limits and excluded tender types.",
                    "After recovery, submit queued transactions once, reconcile every receipt and authorization, and investigate possible duplicates before closing the incident.",
                ],
                [
                    (("one register", "all register", "every register", "multiple register"), "Is one register affected or every register?"),
                    (("network", "handheld", "online", "status page"), "Are the store network, handhelds, and provider status page reachable?"),
                    (("retry", "duplicate", "charged"), "Has any card been retried or possibly charged more than once?"),
                    (("approved", "incident owner", "offline"), "Has the current incident owner approved offline processing for this outage?"),
                ],
            )
        if self._contains_signal(lowered, ("return*", "refund*", "receipt*", "damaged item*", "exchange*")):
            return (
                "return or customer exception",
                [
                    "Verify the current purchase independently using the transaction, loyalty record, item identity, date, price, and original tender before deciding on an exception.",
                    "Inspect and document the current item condition, packaging, reason for return, and whether the facts fit normal policy or require manager approval.",
                ],
                [
                    "Send any refund only through the currently approved tender path, record the exception reason and approver, and keep damaged merchandise out of saleable inventory.",
                    "Give the customer the confirmed resolution and receipt, then reconcile the inventory or claims record.",
                ],
                [
                    (("date", "day", "week", "month"), "When was the item purchased, and is it inside the current return window?"),
                    (("loyalty", "transaction", "receipt", "card"), "What independent proof links this customer and item to the original transaction?"),
                    (("damage", "sealed", "used", "condition"), "What is the documented item condition, and is the damage consistent with the customer’s explanation?"),
                    (("manager", "approval", "exception"), "Does this resolution require a manager or policy-owner exception?"),
                ],
            )
        if self._contains_signal(lowered, ("inventory", "markdown*", "clearance", "sell-through", "weeks of supply", "floor reset", "seasonal")):
            return (
                "inventory or markdown decision",
                [
                    "Freeze the current SKU and store scope, validate on-hand inventory and open orders, and calculate current sell-through, weeks of supply, timing, and margin exposure.",
                    "Segment locations by current demand instead of applying one historical markdown or transfer decision everywhere.",
                ],
                [
                    "Use a limited, measurable first action when timing permits, then recheck units, margin dollars, and aged inventory before expanding it.",
                    "Record the final decision by SKU and location, including exceptions, customer reservations, and the owner who approved it.",
                ],
                [
                    (("unit", "quantity", "inventory"), "How many units are affected, and has on-hand inventory been reconciled?"),
                    (("day", "week", "reset", "deadline"), "How much selling time remains before the current reset or deadline?"),
                    (("sell-through", "weeks of supply", "margin"), "What are current sell-through, weeks of supply, and margin constraints by location?"),
                    (("transfer", "freight", "store"), "Could a transfer preserve more margin than a markdown after freight and handling?"),
                ],
            )
        if self._contains_signal(lowered, ("staffing", "labor", "schedule*", "checkout line*", "traffic", "payroll", "shift*")):
            return (
                "staffing or labor decision",
                [
                    "Measure the current demand pattern by location and time window, including traffic, wait time, conversion, coverage requirements, and available labor hours.",
                    "Move hours from verified low-demand windows into the current peak period before requesting more payroll.",
                ],
                [
                    "Pilot the schedule change in a limited set of representative locations and keep total labor, breaks, safety coverage, and local requirements within policy.",
                    "Compare wait time, conversion, employee coverage, and payroll after the pilot before expanding the schedule.",
                ],
                [
                    (("store", "location"), "Which stores or locations show the problem?"),
                    (("day", "time", "hour", "weekend", "morning", "evening"), "Which exact time windows are overstaffed and understaffed?"),
                    (("wait", "traffic", "conversion", "coverage"), "What current traffic, wait-time, conversion, and coverage evidence supports the change?"),
                    (("budget", "payroll", "hours"), "Must total payroll remain fixed, or is additional labor available?"),
                ],
            )
        if self._contains_signal(lowered, ("presentation*", "powerpoint", "slide*", "deck*", "pitch*", "investor*")):
            return (
                "presentation",
                [
                    "Define the current audience, the decision or investment being requested, and the one message they must remember.",
                    "Build today’s story around the audience’s problem, the product response, verified evidence, business model, differentiation, and exact ask rather than copying the old deck.",
                ],
                [
                    "Verify every current claim, number, source, and date; move supporting detail to notes or an appendix.",
                    "End with the current owner, amount or decision requested, use of resources, milestones, and next date, then rehearse against the actual time limit.",
                ],
                [
                    (("investor", "board", "executive", "client", "audience"), "Who is the current audience?"),
                    (("raise", "funding", "amount", "decision", "ask"), "What exact decision, funding amount, or commitment are you asking for?"),
                    (("minute", "time", "slide"), "How much time and how many slides are available?"),
                    (("traction", "revenue", "customer", "growth", "evidence"), "Which current traction or outcome claims can be verified?"),
                ],
            )
        if self._contains_signal(lowered, ("report", "dashboard*", "metric*", "csv", "kpi*", "retention")):
            return (
                "reporting or analytics",
                [
                    "Freeze the current question, date range, timezone, filters, account scope, and metric definition before changing a report or choosing a number.",
                    "Reconcile one current sample from source to output and separate confirmed discrepancies from possible explanations.",
                ],
                [
                    "Present the current finding in the audience’s language with a visible definition, source, date, limitation, and one clear takeaway.",
                    "Assign an owner to resolve any conflicting definition or unexplained data difference before publication.",
                ],
                [
                    (("date", "range", "week", "month", "quarter"), "What exact reporting period is being used?"),
                    (("timezone", "filter", "segment", "account"), "Which timezone, filters, segments, and account scope apply?"),
                    (("definition", "formula", "metric"), "What is the approved definition or formula for the metric?"),
                    (("source", "warehouse", "dashboard", "export"), "Which source should be authoritative for this decision?"),
                ],
            )
        if self._contains_signal(lowered, ("security", "secret*", "key*", "privacy", "delete*", "production access", "breach*")):
            return (
                "security or sensitive-data decision",
                [
                    "Stop any irreversible or privilege-expanding action and identify the current tenant, system, scope, owner, and time involved.",
                    "Minimize sensitive information in notes and preserve only the evidence required by the approved security or privacy process.",
                ],
                [
                    "Use the current authorized owner and workflow before rotating, deleting, exporting, or changing production access.",
                    "Verify containment, approvals, audit records, and follow-up ownership before closing the case.",
                ],
                [
                    (("tenant", "system", "account"), "Which tenant, system, or account is affected?"),
                    (("time", "when", "timestamp"), "When did the exposure or request begin?"),
                    (("security", "privacy", "owner"), "Which security or privacy owner has been notified?"),
                    (("contain", "revoke", "restrict"), "What containment has already been completed without expanding exposure?"),
                ],
            )
        if self._contains_signal(lowered, ("code", "python", "javascript", "api", "error*", "bug*", "crash*", "fail*")):
            return (
                "software or code issue",
                [
                    "Turn the current report into the smallest reproducible example with exact input, expected result, actual result, version, timestamp, and environment.",
                    "Reproduce it before changing code, preserve the complete sanitized error or request, and compare the failing path with the last known-good path.",
                ],
                [
                    "Apply the narrowest reversible change and add a regression test for this exact current failure.",
                    "Verify normal, missing, null, boundary, retry, and compatibility cases before release or handoff.",
                ],
                [
                    (("error", "traceback", "status", "message"), "What is the complete current error, status code, or traceback?"),
                    (("version", "release", "deployment", "commit"), "Which version is failing, and what is the last known-good version?"),
                    (("input", "request", "payload"), "What smallest current input still reproduces the problem?"),
                    (("expected", "actual"), "What exact result was expected and what happened instead?"),
                ],
            )
        if self._contains_signal(lowered, ("client*", "customer*", "tenant*", "user*")):
            return (
                "customer or client issue",
                [
                    "Confirm the current customer, affected users, exact impact, timeline, environment, and what has already been tried.",
                    "Verify the current behavior with a low-risk test before promising that the historical resolution applies.",
                ],
                [
                    "Give the customer confirmed facts, the next current action, the owner, and a concrete update time without promising an unverified fix.",
                    "Escalate with a concise evidence package if the issue repeats, affects multiple customers, or requires an exception.",
                ],
                [
                    (("tenant", "account", "customer", "client"), "Which customer account or tenant is affected?"),
                    (("user", "people", "everyone", "multiple"), "How many users are affected and what can they not do?"),
                    (("time", "when", "since", "after"), "When did the behavior begin and what changed beforehand?"),
                    (("tried", "tested", "attempted"), "What has already been tried, and what happened?"),
                ],
            )
        return (
            "general company task",
            [
                "Define the current outcome, constraints, people affected, evidence available, and what would count as success.",
                "Use the closest historical decision as a method, not as proof that the same outcome is correct today.",
            ],
            [
                "Take the smallest reversible action, record the result, and compare it with a specific success condition.",
                "Request human review when evidence remains weak or the decision affects safety, money, privacy, customers, or production.",
            ],
            [
                (("outcome", "goal", "result"), "What exact result is needed?"),
                (("constraint", "budget", "deadline", "policy"), "What constraints, deadlines, or policies limit the options?"),
                (("tried", "tested", "attempted"), "What has already been tried, and what happened?"),
                (("owner", "manager", "approver"), "Who owns or approves the final decision?"),
            ],
        )

    def _historical_actions(self, query: str, result: Prediction, limit: int = 4) -> list[tuple[str, SimilarCase]]:
        """Select the most useful individual actions across retrieved decisions."""
        if result.knowledge_match < 0.15:
            return []
        query_domain = self._answer_profile(query)[0]
        records: list[tuple[str, SimilarCase, float, int, int, bool]] = []
        for case_rank, case in enumerate(result.similar_cases):
            case_text = " ".join((case.summary, case.reasoning, case.instructions, case.methods))
            same_domain = self._answer_profile(case_text)[0] == query_domain
            for step_rank, action in enumerate(self._instruction_lines(case.instructions or case.reasoning)):
                records.append((action, case, 0.0, case_rank, step_rank, same_domain))
        if not records:
            return []

        query_vector = self.vectorizer.transform([query])
        action_vectors = self.vectorizer.transform([record[0] for record in records])
        action_scores = cosine_similarity(query_vector, action_vectors)[0]
        scored = []
        for index, (action, case, _, case_rank, step_rank, same_domain) in enumerate(records):
            action_similarity = float(action_scores[index])
            # Do not carry an operational step across unrelated domains merely
            # because it happened to come from the nearest low-quality match.
            if query_domain != "general company task" and not same_domain:
                continue
            if not same_domain and action_similarity < 0.12:
                continue
            if same_domain and action_similarity < 0.01 and case.similarity < 0.25:
                continue
            score = action_similarity + (case.similarity * 0.45) + (0.12 / (case_rank + 1)) + (0.03 / (step_rank + 1))
            scored.append((score, action, case))
        scored.sort(key=lambda item: item[0], reverse=True)

        selected: list[tuple[str, SimilarCase]] = []
        seen: list[set[str]] = []
        for _, action, case in scored:
            words = set(re.findall(r"[a-zA-Z]{4,}", action.lower()))
            if any(len(words & prior) / max(1, len(words | prior)) > 0.62 for prior in seen):
                continue
            selected.append((action, case))
            seen.append(words)
            if len(selected) == limit:
                break
        return selected

    def _adapt_historical_action(self, action: str, query: str, case: SimilarCase) -> str:
        """Make a historical instruction explicitly conditional on today's facts."""
        cleaned = re.sub(r"\b(?:NC|SF|POS|RET|WO)-[A-Z0-9-]+\b", "the current case identifier", action, flags=re.IGNORECASE)
        cleaned = cleaned[0].lower() + cleaned[1:] if cleaned else cleaned
        current_values = re.findall(
            r"\b\$?\d+(?:\.\d+)?\s*(?:%|mb|gb|kb|units?|users?|customers?|hours?|days?|minutes?|seconds?|°f|°c)?\b",
            query,
            flags=re.IGNORECASE,
        )
        value_note = (
            f" Interpret this step using today’s stated value(s)—{', '.join(dict.fromkeys(current_values))}. If the historical numbers bracket a boundary, recalculate equivalent just-below and just-above tests instead of copying them blindly."
            if current_values and re.search(r"\d", action)
            else ""
        )
        method = case.methods.split(";")[0].strip() if case.methods else "the expert’s verification method"
        return f"Apply {method} to the current request: {cleaned}{value_note} Source basis: {case.source_file}."

    def _missing_context_questions(self, query: str) -> list[str]:
        lowered = query.lower()
        _, _, _, possible = self._answer_profile(query)
        questions = [
            question
            for keywords, question in possible
            if not any(keyword in lowered for keyword in keywords)
        ]
        return questions[:4]

    def _direct_answer(self, query: str, result: Prediction) -> str:
        closest = result.similar_cases[0] if result.similar_cases and result.knowledge_match >= 0.15 else None
        if result.knowledge_match >= 0.45:
            strength = "This is a strong historical match, so use the same decision pattern while substituting today’s scope, values, systems, owners, and success checks."
        elif result.knowledge_match >= 0.20:
            strength = "This is a partial historical match. Carry forward the expert’s method and escalation logic, but verify the outcome instead of copying the old resolution."
        elif result.knowledge_match < 0.15:
            strength = "The retrieved company records do not support a recommendation for this request. The actions below are general guidance, not adaptations of unrelated company decisions."
        else:
            strength = "No past decision fully answers this request. SkillVault is using the closest decision only as a reasoning pattern and combining it with a cautious current-case plan."
        anchor = (
            f"The closest expert chose {closest.label.lower()} because {closest.reasoning.rstrip('.')}. "
            if closest else
            ""
        )
        if result.knowledge_match < 0.15:
            return strength
        return f"{strength} {anchor}The concrete recommendation below applies that judgment without treating the historical label as the answer."

    def _concrete_recommendation(self, query: str, result: Prediction) -> str:
        """Lead with actions the employee can take, even when review is required."""
        _, opening, closing, _ = self._answer_profile(query)
        immediate = opening[0].rstrip(".")
        verification = closing[0]
        selected = self._historical_actions(query, result, limit=1)
        historical_method = ""
        if selected:
            action, case = selected[0]
            historical_method = (
                f" Then adapt this proven company action to the current facts: "
                f"{action.rstrip('.')} [Source: {case.source_file}]."
            )
        return f"{immediate}.{historical_method} {verification}"

    def _independent_recommendations(self, query: str, result: Prediction) -> list[str]:
        """Offer useful non-policy ideas without pretending they came from company records."""
        domain, _, _, _ = self._answer_profile(query)
        domain_guidance = {
            "presentation": [
                "Use one decision-relevant claim per slide, put supporting detail in notes or an appendix, and rehearse once against the real time limit.",
                "For market, competitor, or investor facts that may have changed, check current primary sources and record the source date before adding the claim.",
            ],
            "software or code issue": [
                "For version-dependent behavior, check the current official vendor documentation, release notes, known-issue tracker, and status page using sanitized search terms.",
                "Add a regression test before the fix and use a narrow rollout or feature flag when the current environment supports one.",
            ],
            "reporting or analytics": [
                "Create a one-row reconciliation from the authoritative source through every transformation to the final displayed value.",
                "If an external benchmark is needed, use a current primary dataset and record its publication date, definition, and limitations.",
            ],
            "customer or client issue": [
                "Check the current official product status page, release notes, and known issues with customer identifiers removed before deciding the problem is unique.",
                "Send a factual update with impact, action underway, owner, and next-update time even when the final cause is not known yet.",
            ],
            "security or sensitive-data decision": [
                "Use only approved internal policy and authoritative vendor or regulatory sources; do not paste incident details, credentials, or customer data into public search tools.",
                "Prefer reversible containment and evidence preservation before changing access, deleting data, or rotating shared production credentials.",
            ],
            "safety incident": [
                "Do not use general web results to determine safety disposition; use the company’s approved safety process and current authoritative regulatory guidance.",
                "Keep containment in place until measurements, affected scope, corrective action, and authorized release are documented.",
            ],
        }
        defaults = [
            "Take the smallest reversible action first, define the success check before acting, and record what changed.",
            "When a current external fact matters, research only authoritative primary sources with private names, identifiers, and incident details removed.",
        ]
        # Keep at least two domain-specific recommendations visible instead
        # of allowing generic safeguards to crowd them out.
        combined = [*result.general_suggestions[:2], *domain_guidance.get(domain, defaults)]
        unique: list[str] = []
        seen: set[str] = set()
        for suggestion in combined:
            normalized = re.sub(r"\W+", " ", suggestion.lower()).strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique.append(suggestion)
        return unique[:4]

    def _current_case_mapping(self, query: str, result: Prediction) -> str:
        closest = result.similar_cases[0] if result.similar_cases and result.knowledge_match >= 0.15 else None
        current = re.sub(r"\s+", " ", query).strip()[:360]
        if not closest:
            return f"- **Today’s request:** {current}\n- **Historical basis:** No applicable approved decision was found. The closest records are shown separately as historical evidence."
        unique_terms = self._important_query_terms(query, closest.summary)
        differences = (
            ", ".join(unique_terms)
            if unique_terms else
            "The wording is close, but current values, owners, dates, and operating conditions still require verification."
        )
        return (
            f"- **Today’s request:** {current}\n"
            f"- **Closest past decision:** {closest.summary}\n"
            f"- **Judgment being reused:** {closest.reasoning}\n"
            f"- **Current facts not established by that decision:** {differences}"
        )

    def _build_adapted_plan(self, query: str, result: Prediction) -> str:
        domain, opening, closing, _ = self._answer_profile(query)
        historical = self._historical_actions(query, result)
        steps = [opening[0]]
        if domain == "presentation":
            duration_match = re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)[ -]?(?:minutes?|mins?)\b", query, flags=re.IGNORECASE)
            if duration_match:
                raw_minutes = duration_match.group(1).lower()
                minute_words = {word: number for number, word in enumerate(("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"))}
                minutes = int(raw_minutes) if raw_minutes.isdigit() else minute_words[raw_minutes]
                if 1 <= minutes <= 10:
                    steps.append(
                        f"Adapt the deck to a {minutes}-minute presentation: open with the audience's problem, show the most relevant verified evidence, explain the solution briefly, and end with a specific ask. Move architecture and supporting detail to the appendix. Rehearse until the spoken version fits {minutes} minutes."
                    )
        if result.knowledge_match < 0.35:
            if result.knowledge_match < 0.15:
                steps.append("Treat the retrieved company decisions as insufficient for this task. Use general steps only, and seek an applicable expert source before taking specialized or irreversible action.")
            else:
                steps.append(opening[1])
        steps.extend(self._adapt_historical_action(action, query, case) for action, case in historical)
        if result.knowledge_match >= 0.35 and len(opening) > 1:
            steps.insert(1, opening[1])
        steps.extend(closing)
        steps.append(
            "Write down the confirmed result, the success check used, any remaining risk, and the next owner so the outcome can improve the company’s future decision record."
        )

        deduplicated = []
        normalized_seen = set()
        for step in steps:
            normalized = " ".join(re.findall(r"[a-zA-Z]{4,}", step.lower())[:12])
            if normalized and normalized not in normalized_seen:
                normalized_seen.add(normalized)
                deduplicated.append(step)
        return "\n\n".join(f"{index}. {step}" for index, step in enumerate(deduplicated[:8], start=1))

    def _draft_for_current_case(self, query: str, result: Prediction) -> tuple[str, str]:
        domain, _, _, _ = self._answer_profile(query)
        if domain == "presentation":
            return (
                "#### Current deliverable structure",
                "1. **Audience and decision** — name who is deciding and the exact ask.\n"
                "2. **Problem and urgency** — show who has the problem and why it matters now.\n"
                "3. **Solution and demonstration** — show the smallest convincing workflow.\n"
                "4. **Verified evidence** — use current traction, outcomes, dates, and sources.\n"
                "5. **Business and differentiation** — explain market, model, and why this approach wins.\n"
                "6. **Ask and milestones** — state the amount or decision, use of resources, owner, and next date.",
            )
        if domain == "reporting or analytics":
            return (
                "#### Current report structure",
                "1. **Decision or takeaway**\n2. **Metric definition and scope**\n3. **Verified evidence and source**\n"
                "4. **Confirmed interpretation versus hypotheses**\n5. **Action, owner, and deadline**",
            )
        if domain == "software or code issue":
            return (
                "#### Current engineering handoff",
                f"**Issue:** {query.strip()}\n\n"
                "**Recommended path:** reproduce the failure, isolate the smallest cause, make the narrowest reversible change, add a regression test, and verify it before release.\n\n"
                "**Include:** exact reproduction, expected versus actual behavior, environment/version, timestamps, sanitized evidence, recent changes, impact, test result, and remaining unknowns.",
            )
        if domain in ("safety incident", "payment or POS incident"):
            return (
                "#### Current incident handoff",
                f"**Issue:** {query.strip()}\n\n"
                "**Include:** current location/system, first observed time, scope, containment already completed, evidence captured, customer impact, owner contacted, approvals needed, and the next verification checkpoint.",
            )
        return (
            "#### Current decision record",
            f"**Request:** {query.strip()}\n\n"
            "**Recommended path:** execute the adapted reversible steps above, verify the stated success condition, and record any change from the historical assumptions.\n\n"
            "**Record:** confirmed facts, assumptions, retrieved source, adapted actions, approver, success check, outcome, and the next review date.",
        )

    def _legacy_compose_response(self, query: str, result: Prediction) -> str:
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

    def compose_response(self, query: str, result: Prediction, feedback_profile: dict | None = None) -> str:
        """Build a current-case answer that adapts retrieved expert judgment at every match level."""
        evidence = result.similar_cases
        selected_actions = self._historical_actions(query, result)
        closest = evidence[0] if evidence and result.knowledge_match >= 0.15 else None
        used_cases = []
        if closest is not None:
            used_cases.append(closest)
        for _, case in selected_actions:
            if case not in used_cases:
                used_cases.append(case)
        action_cases = list(dict.fromkeys(case.source_file for _, case in selected_actions))
        tools = list(
            dict.fromkeys(
                case.software
                for case in used_cases
                if case.software and case.source_file in action_cases
            )
        )
        methods = list(dict.fromkeys(case.methods for case in used_cases if case.methods))
        sources = list(dict.fromkeys(case.source_file for case in used_cases if case.source_file))
        domain, _, _, _ = self._answer_profile(query)
        weak_match = result.knowledge_match < 0.35

        mapping = self._current_case_mapping(query, result)
        recommendation = self._concrete_recommendation(query, result)
        plan = self._build_adapted_plan(query, result)
        direct_answer = self._direct_answer(query, result)
        questions = self._missing_context_questions(query)
        question_text = (
            "\n".join(f"- {question}" for question in questions)
            if questions else
            "- No obvious core detail is missing, but confirm current owners, values, policies, and success criteria before acting."
        )

        if domain in ("safety incident", "security or sensitive-data decision"):
            escalation = "Use the adapted containment steps now, but the designated safety, security, privacy, or incident owner must approve the final disposition."
        elif weak_match:
            escalation = "The retrieved decision is a weak match. Use this as a provisional plan, collect the missing context above, and have the responsible expert approve any irreversible, customer-facing, financial, or production change."
        else:
            escalation = "Proceed with the adapted plan while checking each historical assumption against the current case. Escalate if an owner, threshold, policy, or risk differs from the retrieved decision."

        tool_text = "; ".join(tools) if tools else "No specific software was recorded."
        method_text = "; ".join(methods) if methods else "Use the evidence and verification steps in the adapted plan."
        source_text = "\n".join(f"- **{source}**" for source in sources) or "- No source file was recorded."
        draft_title, draft_body = self._draft_for_current_case(query, result)
        independent = self._independent_recommendations(query, result)
        independent_text = "\n".join(f"- {item}" for item in independent)
        if result.knowledge_match < 0.15:
            workflow_note = "*No applicable historical workflow category was found for this question.*"
            overlap_note = (
                f"The closest historical similarity is **{result.knowledge_match:.0%}**. "
                "It measures retrieval overlap, so the general plan below does not use the unrelated record as a source."
            )
        else:
            workflow_note = f"*Historical workflow category used for routing—not the recommendation itself: {result.label}.*"
            overlap_note = (
                f"The displayed historical similarity is **{result.knowledge_match:.0%}**. "
                "That percentage measures retrieval overlap; the plan below is still rewritten for today’s request rather than copied from the old record."
            )

        from .recommendations import two_recommendations

        adapted = f"""### Recommendation

{recommendation}

{workflow_note}

{direct_answer}

{overlap_note}

#### How the past decision maps to today

{mapping}

#### Adapted action plan for this request

{plan}

#### Independent recommendations (not company policy)

These are practical safeguards added beyond the retrieved company records. SkillVault did not perform a live internet search:

{independent_text}

#### Details to confirm

{question_text}

#### Tools and methods carried forward

- **Recorded tools:** {tool_text}
- **Recorded methods:** {method_text}
- Use a recorded tool only if it is available and approved in the current environment.

#### When to stop or get approval

{escalation}

{draft_title}

{draft_body}

#### Company evidence used

{source_text}
"""
        rules = set((feedback_profile or {}).get("rules", []))
        if rules & {"specific", "steps", "detail", "clear", "recheck"}:
            additions = ["#### Extra execution detail requested through saved feedback",
                         "This is rules-based guidance, not a newly verified company procedure."]
            if rules & {"specific", "detail", "steps"}:
                additions += [f"**Working artifact for this request:**\n{draft_body}",
                              "**Before applying it:** resolve the following current-case details:\n" + question_text,
                              "**After each change:** record the expected result, the actual result, and the evidence. If they differ, stop that step and revisit its assumptions before expanding the change."]
            if "clear" in rules:
                additions.append("**How to read the plan:** company evidence records what happened before; adapted actions are proposals for this request; independent advice is not company policy.")
            if "recheck" in rules:
                additions.append("**Similar advice previously received negative feedback.** The fallback cannot establish a new fix from a rating. Recheck the applicability mapping below and answer the missing-context questions before repeating the previous approach.\n" + mapping)
            adapted += "\n\n" + "\n\n".join(additions)
        if "concise" in rules:
            adapted = f"### Recommendation\n{recommendation}\n\n### Actions\n{plan}\n\n### Independent guidance—not company policy\n{independent_text}\n\n### Confirm before acting\n{question_text}\n\n### Approval boundary\n{escalation}\n\n### Sources\n{source_text}"
        return two_recommendations(result, adapted, fallback=True)

    def build_decision_lineage(
        self,
        query: str,
        result: Prediction,
        approval_status: str = "Awaiting approval",
        plan_id: str = "",
    ) -> dict[str, object]:
        """Create an auditable record of how evidence became a current decision."""
        closest = result.similar_cases[0] if result.similar_cases else None
        selected_actions = self._historical_actions(query, result)
        current_domain = self._answer_profile(query)[0]
        quantities = list(
            dict.fromkeys(
                re.findall(
                    r"\b\$?\d+(?:\.\d+)?\s*(?:%|mb|gb|kb|units?|users?|customers?|hours?|days?|minutes?|seconds?|°f|°c)?\b",
                    query,
                    flags=re.IGNORECASE,
                )
            )
        )
        current_only_terms = self._important_query_terms(query, closest.summary if closest else "")

        changes = [
            f"Reframed the historical process for the current {current_domain}.",
            "Required current scope, owners, systems, policies, and success checks instead of assuming the old case is identical.",
        ]
        if quantities:
            changes.append(
                f"Used the current value(s) {', '.join(quantities)} as today’s evidence and treated historical values as context, not automatic thresholds."
            )
        if current_only_terms:
            changes.append(
                f"Added current details not established by the past decision: {', '.join(current_only_terms)}."
            )
        if result.knowledge_match < 0.35:
            changes.append(
                "Restricted the weak historical match to its general judgment and excluded operational steps from unrelated work domains."
            )

        kept_actions = [
            {
                "action": action,
                "source_file": case.source_file,
                "method": case.methods,
            }
            for action, case in selected_actions
        ]
        if not kept_actions and closest is not None:
            kept_actions.append(
                {
                    "action": closest.reasoning,
                    "source_file": closest.source_file,
                    "method": closest.methods,
                }
            )

        return {
            "plan_id": plan_id,
            "current_problem": re.sub(r"\s+", " ", query).strip(),
            "retrieval": {
                "match_strength": (
                    "Strong" if result.knowledge_match >= 0.45
                    else "Partial" if result.knowledge_match >= 0.20
                    else "Weak"
                ),
                "similarity": result.knowledge_match,
                "retrieval_mode": result.retrieval_mode,
                "candidate_count": result.candidate_count,
                "indexed_count": result.indexed_count,
            },
            "retrieved_decisions": [
                {
                    "summary": case.summary,
                    "source_file": case.source_file,
                    "similarity": case.similarity,
                    "expert_approach": case.label,
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
                for case in result.similar_cases
            ],
            "judgment_kept": kept_actions,
            "changes_for_today": changes,
            "missing_context": self._missing_context_questions(query),
            "final_action": {
                "recommended_approach": result.label,
                "adapted_plan": self._build_adapted_plan(query, result),
                "approval_status": approval_status,
            },
            "outcome": {
                "status": "Pending execution",
                "summary": "",
            },
        }

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
        return pd.DataFrame(
            [
                {
                    "Case": i + 1,
                    "Expert approach": case.label,
                    "Scenario": case.summary,
                    "Owner": case.expert_owner,
                    "Department": case.department,
                    "Approval date": case.approval_date,
                    "Last reviewed": case.last_reviewed_date,
                    "Expiration date": case.expiration_date,
                    "Status": case.lifecycle_status,
                    "Goal": case.goal,
                    "Chosen approach": case.chosen_approach,
                    "Alternatives considered": case.alternatives_considered,
                    "Constraints": case.constraints,
                    "Reusable rule": case.reusable_rule,
                    "Exceptions": case.exceptions,
                    "Software/tools": case.software,
                    "Methods": case.methods,
                    "Source file": case.source_file,
                    "Outcome": case.outcome,
                    "Reasoning": case.reasoning,
                }
                for i, case in enumerate(self.cases)
            ]
        )
