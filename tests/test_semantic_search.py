from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from skillvault.data import ExpertCase, infer_features
from skillvault.engine import SkillVaultEngine
from skillvault.semantic_search import SemanticIndex, SemanticSearchError


class SemanticTests(unittest.TestCase):
    def test_semantic_candidate_incremental_persistence_and_fallback(self):
        cases = [
            ExpertCase("Demonstrate repeatable customer adoption", "Follow standard process", "Show traction before architecture", infer_features("presentation"), source_file="pitch.md"),
            ExpertCase("Fix slow database queries", "Diagnose and verify", "Inspect database query plans", infer_features("code"), source_file="db.md"),
        ]
        calls = []
        def embed(texts):
            calls.extend(texts)
            return [[1, 0] if "adoption" in text or "convince backers" in text else [0, 1] for text in texts]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "a.sqlite3"
            index = SemanticIndex(path, embed=embed)
            self.assertEqual(index.build(cases), 2)
            self.assertEqual(index.build(cases), 0)
            self.assertEqual(len(calls), 2)
            engine = SkillVaultEngine(cases, semantic_index=index)
            engine.train()
            result = engine.predict("How can I convince backers people want this?")
            self.assertEqual(result.similar_cases[0].source_file, "pitch.md")
            self.assertIn("semantic", result.retrieval_mode)
            isolated = SemanticIndex(Path(directory) / "b.sqlite3", embed=embed)
            self.assertIsNone(isolated.load(cases)[1])
            cases[0].reasoning = "Changed current method"
            self.assertEqual(index.build(cases), 1)
            def unavailable(texts):
                raise OSError("offline")
            index._embed_override = unavailable
            self.assertIn("Keyword", engine.predict("database queries").retrieval_mode)

    def test_malformed_embeddings_are_rejected(self):
        with TemporaryDirectory() as directory:
            index = SemanticIndex(Path(directory) / "index.sqlite3", embed=lambda texts: [[float("nan"), 0]])
            with self.assertRaises(SemanticSearchError):
                index.embed(["question"])
