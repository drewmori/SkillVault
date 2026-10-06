from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from skillvault.data import ExpertCase, infer_features
from skillvault.engine import SkillVaultEngine
from skillvault.importer import DecisionDraft, WorkFile, draft_from_completed_work
from skillvault.local_model import DECISION_SCHEMA, LocalModelError, SkillVaultLocalModel
from skillvault.capture import capture_work
from skillvault.storage import (
    case_from_dict,
    case_to_dict,
    load_recommendation_feedback,
    save_recommendation_feedback,
)


class FakeOllamaTransport:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls: list[tuple[str, str, dict[str, object] | None, float]] = []

    def __call__(
        self,
        method: str,
        path: str,
        payload: dict[str, object] | None,
        timeout: float,
    ) -> dict[str, object]:
        self.calls.append((method, path, payload, timeout))
        if path == "/api/tags":
            return {"models": [{"name": "qwen3.5:9b", "model": "qwen3.5:9b"}]}
        return {"message": {"role": "assistant", "content": self.outputs.pop(0)}}


class DecisionPipelineTests(unittest.TestCase):
    def test_recommendation_feedback_is_persisted_without_becoming_knowledge(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            feedback_path = Path(temporary_directory) / "recommendation_feedback.json"
            feedback_id = save_recommendation_feedback(
                feedback_path,
                {
                    "query": "Uploads fail above 100 MB",
                    "verdict": "Needs improvement",
                    "issue_types": ["Missing important steps"],
                    "suggested_steps": "Compare gateway and server limits.",
                    "expert_approved": False,
                },
            )

            stored = load_recommendation_feedback(feedback_path)

            self.assertTrue(feedback_id.startswith("feedback-"))
            self.assertEqual(len(stored), 1)
            self.assertEqual(stored[0]["feedback_id"], feedback_id)
            self.assertFalse(stored[0]["expert_approved"])
            self.assertIn("recorded_at", stored[0])

    def test_completed_work_extracts_structured_decision_context(self) -> None:
        artifact = WorkFile(
            "investor_deck_notes.md",
            "text/markdown",
            (
                "Situation: Seed investor pitch needed restructuring\n"
                "Goal: Prove adoption is repeatable\n"
                "Chosen approach: Lead with customer proof and move architecture to the appendix\n"
                "Decision rationale: Investors had eight minutes and prioritized traction over implementation detail\n"
                "Alternatives considered: Opening with a product tour was rejected\n"
                "Constraints: Eight-minute pitch; verified claims only\n"
                "1. Open with the customer problem\n"
                "2. Show verified adoption evidence\n"
                "Outcome: Investors requested a technical follow-up\n"
                "Reusable rule: Lead investor decks with proof before architecture\n"
                "Exceptions: Use a technical opening for an engineering diligence session\n"
            ).encode("utf-8"),
        )

        draft = draft_from_completed_work([artifact])

        self.assertEqual(draft.goal, "Prove adoption is repeatable")
        self.assertIn("customer proof", draft.chosen_approach)
        self.assertIn("product tour", draft.alternatives_considered)
        self.assertIn("Eight-minute pitch", draft.constraints)
        self.assertIn("proof before architecture", draft.reusable_rule)
        self.assertEqual(draft.missing_context, [])

    def test_structured_fields_survive_storage_round_trip(self) -> None:
        case = ExpertCase(
            "Investor deck needed a stronger opening",
            "Follow standard process",
            "The audience needed proof before detail.",
            infer_features("investor presentation"),
            "investor_deck.pptx",
            "1. Lead with proof.\n2. Move architecture to the appendix.",
            "PowerPoint",
            "Audience-first storytelling",
            "Follow-up meeting booked",
            goal="Win a diligence meeting",
            chosen_approach="Lead with traction",
            alternatives_considered="Product tour first",
            constraints="Eight minutes",
            reusable_rule="Proof before detail",
            exceptions="Technical diligence meetings",
        )

        restored = case_from_dict(case_to_dict(case))

        self.assertEqual(restored.goal, case.goal)
        self.assertEqual(restored.chosen_approach, case.chosen_approach)
        self.assertEqual(restored.alternatives_considered, case.alternatives_considered)
        self.assertEqual(restored.constraints, case.constraints)
        self.assertEqual(restored.reusable_rule, case.reusable_rule)
        self.assertEqual(restored.exceptions, case.exceptions)

    def test_local_model_extraction_is_structured_and_can_receive_images(self) -> None:
        structured = {
            "summary": "Investor deck needed restructuring",
            "decision": "Follow standard process",
            "goal": "Prove repeatable adoption",
            "chosen_approach": "Lead with customer proof",
            "reasoning": "The audience prioritized traction",
            "alternatives_considered": ["Product tour first"],
            "constraints": ["Eight-minute pitch"],
            "instructions": ["Lead with the problem", "Show verified proof"],
            "software": ["PowerPoint"],
            "methods": ["Audience-first storytelling"],
            "outcome": "Technical follow-up requested",
            "reusable_rule": "Proof before architecture",
            "exceptions": ["Technical diligence session"],
            "missing_context": [],
        }
        transport = FakeOllamaTransport([json.dumps(structured)])
        backend = SkillVaultLocalModel(transport=transport)
        draft = DecisionDraft(
            "Deck",
            "Follow standard process",
            "",
            "",
            "",
            "",
            "",
            "deck.pptx; slide.png",
            evidence_preview="Slide 1: customer proof",
            media=[{"name": "slide.png", "type": "image/png", "bytes": b"png-bytes"}],
        )

        enriched = backend.enrich_decision_draft(draft, "Presentation or pitch deck")
        call = next(call for call in transport.calls if call[1] == "/api/chat")
        payload = call[2]

        self.assertEqual(payload["model"], "qwen3.5:9b")
        self.assertFalse(payload["stream"])
        self.assertTrue(payload["format"]["additionalProperties"] is False)
        self.assertEqual(enriched.chosen_approach, "Lead with customer proof")
        self.assertTrue(payload["messages"][1]["images"][0])

    def test_model_cannot_turn_missing_rationale_and_outcome_into_claims(self) -> None:
        record = {key: ([] if spec["type"] == "array" else "") for key, spec in DECISION_SCHEMA["properties"].items()}
        record.update(summary="Revised investor slide", decision="Follow standard process",
                      reasoning="Investors preferred the change", outcome="Investors approved the pilot")
        transport = FakeOllamaTransport([json.dumps(record)])
        backend = SkillVaultLocalModel(transport=transport)
        draft = capture_work([], [WorkFile("finished.md", "text/markdown", b"We moved adoption metrics to the first slide. No real-world outcome is supplied.")], [])
        enriched = backend.enrich_decision_draft(draft)
        self.assertIn("expert must add", enriched.reasoning.lower())
        self.assertIn("expert must record", enriched.outcome.lower())
        self.assertTrue(any(any(word in gap.lower() for word in ("rationale", "reason", "why")) for gap in enriched.missing_context))

    def test_grounded_answer_receives_current_question_and_retrieved_evidence(self) -> None:
        transport = FakeOllamaTransport(
            ["What to do now\n1. Lead with verified customer proof. [Source: investor_deck.pptx] [Source: current_deck.md]"]
        )
        backend = SkillVaultLocalModel(transport=transport)
        retrieved = SimpleNamespace(
            summary="Past seed pitch",
            label="Follow standard process",
            source_file="investor_deck.pptx",
            expert_owner="Avery Chen",
            department="Strategy",
            last_reviewed_date="2026-08-01",
            expiration_date="2027-08-01",
            goal="Win a follow-up",
            chosen_approach="Lead with traction",
            reasoning="Investors prioritized evidence",
            alternatives_considered="Product tour first",
            constraints="Eight minutes",
            instructions="Lead with proof",
            software="PowerPoint",
            methods="Audience-first storytelling",
            outcome="Follow-up booked",
            reusable_rule="Proof before detail",
            exceptions="Technical diligence",
        )
        result = SimpleNamespace(
            similar_cases=[retrieved],
            knowledge_match=0.28,
            label="Follow standard process",
        )

        answer = backend.grounded_answer(
            "How should I pitch investors tomorrow?",
            result,
            conversation_history=[{"role": "user", "content": "I have eight minutes."}],
            company_profile={"industry": "Retail", "password": "must-not-be-sent"},
            feedback_profile={"rules": ["detail", "specific"]},
        )
        call = next(call for call in transport.calls if call[1] == "/api/chat")
        payload = json.loads(call[2]["messages"][1]["content"])

        self.assertIn("Lead with verified", answer)
        self.assertNotIn("[Source: current_deck.md]", answer)
        self.assertIn("## 1. Company-data-only recommendation", answer)
        self.assertIn("## 2. Adapted recommendation", answer)
        first_section = answer.split("## 2.", 1)[0]
        self.assertIn("> Lead with proof", first_section)
        self.assertNotIn("Lead with verified", first_section)
        self.assertEqual(payload["current_employee_question"], "How should I pitch investors tomorrow?")
        self.assertEqual(payload["retrieval_match_strength"], "partial")
        self.assertTrue(payload["company_evidence_applicable"])
        self.assertEqual(payload["approved_company_evidence"][0]["source_file"], "investor_deck.pptx")
        self.assertEqual(payload["recent_conversation"][0]["content"], "I have eight minutes.")
        self.assertEqual(payload["company_context"], {"industry": "Retail"})
        self.assertEqual(len(payload["learned_answer_preferences"]), 2)
        self.assertEqual(call[2]["options"]["num_predict"], 3800)
        self.assertTrue(payload["answer_requirements"]["escalation_cannot_be_the_only_recommendation"])

    def test_low_match_fallback_gives_actions_before_escalation(self) -> None:
        case = ExpertCase(
            "API upload failed at a size boundary",
            "Escalate for review",
            "The team isolated the boundary before requesting a production limit change.",
            infer_features("api upload failure"),
            "upload_runbook.md",
            "Reproduce just below and above the failing size.\nCompare the gateway and server limits.\nVerify a small upload still works.",
            "Postman; API gateway logs",
            "Boundary testing; log correlation",
            "The failing gateway limit was identified",
        )
        engine = SkillVaultEngine([case])
        engine.train()
        result = engine.predict("Our new API request fails, but we do not know why.")

        answer = engine.compose_response("Our new API request fails, but we do not know why.", result)

        self.assertIn("### Recommendation", answer)
        self.assertIn("#### Adapted action plan for this request", answer)
        self.assertIn("#### Independent recommendations (not company policy)", answer)
        self.assertLess(answer.index("### Recommendation"), answer.index("#### When to stop or get approval"))
        self.assertIn("official vendor documentation", answer)
        learned = engine.compose_response("Our new API request fails, but we do not know why.", result,
                                          feedback_profile={"rules": ["detail", "specific", "recheck"]})
        self.assertNotEqual(answer, learned)
        self.assertEqual(answer.split("## 2.")[0], learned.split("## 2.")[0])
        self.assertIn("Extra execution detail", learned)

    def test_unreachable_ollama_keeps_local_fallback_available(self) -> None:
        def unavailable_transport(
            method: str,
            path: str,
            payload: dict[str, object] | None,
            timeout: float,
        ) -> dict[str, object]:
            raise LocalModelError("offline")

        backend = SkillVaultLocalModel(transport=unavailable_transport)
        self.assertFalse(backend.available)
        self.assertIn("Ollama", backend.unavailable_reason)


if __name__ == "__main__":
    unittest.main()
