import unittest
from types import SimpleNamespace

from skillvault.recommendations import evidence_only_recommendation, two_recommendations


class RecommendationTests(unittest.TestCase):
    def test_both_match_levels_separate_literal_evidence_from_added_advice(self):
        case = SimpleNamespace(source_file="deck.pptx", instructions="Lead with verified traction.",
                               chosen_approach="Evidence first", constraints="Eight minutes", exceptions="Technical diligence")
        for score in (0.1, 0.8):
            result = SimpleNamespace(similar_cases=[case], knowledge_match=score)
            answer = two_recommendations(result, "AI addition: prepare a financial appendix.")
            first, second = answer.split("## 2.", 1)
            self.assertIn("> Lead with verified traction.", first)
            self.assertIn("deck.pptx", first)
            self.assertIn("Technical diligence", first)
            self.assertNotIn("financial appendix", first)
            self.assertIn("financial appendix", second)
            self.assertEqual("**Weak match:**" in first, score < 0.35)

    def test_no_evidence_does_not_invent_company_recommendation(self):
        result = SimpleNamespace(similar_cases=[], knowledge_match=0)
        self.assertIn("no data-only recommendation available", evidence_only_recommendation(result))
        self.assertIn("rules-based general guidance", two_recommendations(result, "General steps", fallback=True))
