import base64
import unittest
from io import BytesIO
from PIL import Image

from skillvault.importer import WorkFile
from skillvault.local_model import SkillVaultLocalModel, LocalModelError


class VisualComparisonTests(unittest.TestCase):
    def image_file(self, color):
        data = BytesIO()
        Image.new("RGB", (8, 8), color).save(data, format="PNG")
        return WorkFile("screenshot.png", "image/png", data.getvalue())

    def test_images_preserve_before_after_order_and_guardrails(self):
        before, after = self.image_file("red"), self.image_file("blue")
        calls = []
        def transport(method, path, payload, timeout):
            if path == "/api/tags":
                return {"models": [{"name": "qwen3.5:9b"}]}
            calls.append(payload)
            return {"message": {"content": "Observed changes: background changed."}}
        result = SkillVaultLocalModel(transport=transport).compare_work_images(before, after)
        self.assertIn("Observed changes", result)
        self.assertEqual([base64.b64decode(x) for x in calls[0]["messages"][1]["images"]], [before.data, after.data])
        self.assertIn("Do not infer motivation", calls[0]["messages"][0]["content"])

    def test_invalid_images_never_reach_model(self):
        def transport(*args):
            self.fail("Invalid input reached model")
        model = SkillVaultLocalModel(transport=transport)
        for data in (b"not an image", b"", b"x" * 6_000_001):
            with self.assertRaises(LocalModelError):
                model.compare_work_images(WorkFile("bad.png", "image/png", data), self.image_file("white"))
