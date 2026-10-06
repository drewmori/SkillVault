import unittest

from skillvault.current_case import material_followup, prepare_current_case
from skillvault.importer import WorkFile


class CurrentCaseTests(unittest.TestCase):
    def test_employee_document_and_team_are_context_not_company_evidence(self):
        case = prepare_current_case(
            "How should we revise the pitch?", "  Corporate   strategy ",
            WorkFile("current_deck.md", "text/markdown", b"Investor deck for tomorrow. We need to show verified customer traction."),
        )
        self.assertEqual(case.team, "Corporate strategy")
        self.assertIn("current_deck.md", case.model_question())
        self.assertIn("not approved company knowledge", case.model_question())
        self.assertIn("verified customer traction", case.retrieval_question())
        self.assertNotIn("Corporate strategy", case.retrieval_question())

    def test_large_or_unreadable_file_is_not_used(self):
        large = prepare_current_case("Question", file=WorkFile("large.txt", "text/plain", b"x" * (8 * 1024 * 1024 + 1)))
        self.assertEqual(large.excerpt, "")
        self.assertIn("over 8 MB", large.warning)
        unreadable = prepare_current_case("Question", file=WorkFile("image.png", "image/png", b"png"))
        self.assertEqual(unreadable.excerpt, "")
        self.assertIn("No readable text", unreadable.warning)

    def test_one_material_question_only_when_needed(self):
        case = prepare_current_case("Help with this presentation")
        self.assertIn("audience", material_followup(case, 0.6))
        detailed = prepare_current_case("Help with this 5 minute investor presentation")
        self.assertIsNone(material_followup(detailed, 0.7))
        self.assertIsNone(material_followup(prepare_current_case("Explain presentation structure"), 0.1))


if __name__ == "__main__":
    unittest.main()
