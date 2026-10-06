import unittest
import json
from tempfile import TemporaryDirectory
from pathlib import Path
from hashlib import sha256

from skillvault.capture import capture_work, apply_interview, interview_questions, next_interview_question
from skillvault.data import ExpertCase, infer_features
from skillvault.engine import SkillVaultEngine
from skillvault.importer import WorkFile
from skillvault.storage import case_from_dict, case_to_dict
from skillvault.local_model import _compact_case, SkillVaultLocalModel, LocalModelError, DECISION_SCHEMA
from skillvault.artifacts import store_artifacts, read_artifact


class CaptureTests(unittest.TestCase):
    def test_original_files_are_scoped_and_verified(self):
        with TemporaryDirectory() as directory:
            a, b = Path(directory) / "a", Path(directory) / "b"
            file = WorkFile("../../report.txt", "text/plain", b"original evidence")
            digest = sha256(file.data).hexdigest()
            store_artifacts(a, [file])
            self.assertEqual(read_artifact(a, digest), file.data)
            self.assertIsNone(read_artifact(b, digest))
            self.assertIsNone(read_artifact(a, "../report.txt"))
            (a / digest).write_bytes(b"modified")
            self.assertIsNone(read_artifact(a, digest))

    def test_batch_decisions_keep_independent_fields_and_verified_quotes(self):
        draft = capture_work([], [WorkFile("review.md", "text/markdown", b"Lead with traction.\nUse retry backoff.")], [])
        draft.goal = "A batch-wide goal must not fill missing independent goals"
        def item(summary, quote):
            value = {key: ([] if spec["type"] == "array" else "") for key, spec in DECISION_SCHEMA["properties"].items()}
            value.update(summary=summary, decision="Follow standard process", citations=[{"source_id": "after-1", "quote": quote}])
            return value
        items = [item("Pitch structure", "Lead with traction."), item("API retries", "Use retry backoff.")]
        def transport(method, path, payload, timeout):
            if path == "/api/tags":
                return {"models": [{"name": "qwen3.5:9b"}]}
            return {"message": {"content": json.dumps({"decisions": items})}}
        model = SkillVaultLocalModel(transport=transport)
        decisions = model.split_decision_drafts(draft, "Completed work")
        self.assertEqual(len(decisions), 2)
        self.assertEqual(decisions[0].goal, "")
        self.assertEqual(decisions[1].capture_evidence["decision_citations"][0]["extracted_line"], 2)
        items[1]["citations"][0]["quote"] = "An invented result"
        with self.assertRaises(LocalModelError):
            model.split_decision_drafts(draft, "Completed work")

    def test_before_after_interview_storage_and_retrieval(self):
        before = WorkFile("deck.md", "text/markdown", b"Subject: Investor pitch\n1. Open with technical architecture.\n2. Show product features.")
        after = WorkFile("deck.md", "text/markdown", b"Subject: Investor pitch\n1. Open with customer traction.\n2. Put architecture in the appendix.")
        draft = capture_work([before], [after], [])
        self.assertNotIn("Open with technical architecture", draft.instructions)
        self.assertIn("-1. Open with technical architecture", draft.capture_evidence["comparison"])
        self.assertIn("+1. Open with customer traction", draft.capture_evidence["comparison"])
        self.assertEqual(draft.software, "")
        self.assertIn("reasoning", [key for key, _ in interview_questions(draft, 20)])
        draft = apply_interview(draft, {
            "goal": "Prove repeatable demand", "chosen_approach": "Lead with customer traction",
            "reasoning": "Investors prioritized adoption evidence over architecture.",
            "outcome": "Not measured yet", "instructions": "1. Verify customer traction.\n2. Put the evidence first.",
        })
        self.assertNotIn("reasoning", [key for key, _ in interview_questions(draft, 20)])
        self.assertEqual(draft.capture_evidence["employee_confirmed"]["outcome"], "Not measured yet")
        case = ExpertCase(
            draft.summary, draft.decision, draft.reasoning, infer_features(draft.summary),
            source_file=draft.source_file, instructions=draft.instructions,
            capture_evidence=draft.capture_evidence,
        )
        restored = case_from_dict(case_to_dict(case))
        self.assertEqual(restored.capture_evidence, draft.capture_evidence)
        engine = SkillVaultEngine([restored])
        engine.train()
        result = engine.predict("How do I show investors customer traction?")
        evidence = _compact_case(result.similar_cases[0], "E1")["capture_evidence"]
        self.assertEqual(evidence["source_excerpts"][0]["role"], "before")
        self.assertEqual(evidence["source_excerpts"][1]["role"], "after")
        self.assertIn("adoption evidence", evidence["employee_confirmed"]["reasoning"])

    def test_explanation_only_and_unknown_fields(self):
        draft = capture_work([], [], [], "Situation: Slow checkout during weekend rush")
        self.assertEqual(draft.capture_evidence["documents"], [])
        self.assertTrue(interview_questions(draft))
        revised = apply_interview(draft, {"software": "Invented system", "reasoning": "  Peak demand doubled.  "})
        self.assertEqual(revised.software, "")
        self.assertEqual(revised.reasoning, "Peak demand doubled.")

    def test_no_readable_text_does_not_invent_comparison(self):
        draft = capture_work([WorkFile("old.png", "image/png", b"x")], [WorkFile("new.png", "image/png", b"y")], [])
        self.assertTrue(any("no readable text" in warning for warning in draft.warnings))
        self.assertIn("reasoning", [key for key, _ in interview_questions(draft, 20)])

    def test_explicitly_unmeasured_outcome_is_still_unknown(self):
        draft = capture_work([], [WorkFile("meeting.md", "text/markdown", b"Investor deck revised. No real-world outcome is supplied.")], [])
        self.assertIn("outcome", [key for key, _ in interview_questions(draft, 20)])
        self.assertIn("must record", draft.outcome.lower())

    def test_capture_asks_one_important_gap_at_a_time(self):
        draft = capture_work([], [WorkFile("deck.md", "text/markdown", b"Subject: Investor pitch\nChosen approach: Lead with customer proof\n1. Show verified customer demand")], [])
        self.assertEqual(next_interview_question(draft)[0], "reasoning")
        self.assertIn("Lead with customer proof", next_interview_question(draft)[1])
        revised = apply_interview(draft, {"reasoning": "Investors asked for evidence of demand."})
        self.assertEqual(next_interview_question(revised)[0], "outcome")


if __name__ == "__main__":
    unittest.main()
