from __future__ import annotations

import os
from pathlib import Path
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
from skillvault.capture import capture_work
from skillvault.importer import WorkFile
from io import BytesIO
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


class StreamlitSmokeTests(unittest.TestCase):
    def open_northstar_demo(self, app: AppTest) -> AppTest:
        bypass = next(button for button in app.button if button.label == "Open Northstar demo")
        app = bypass.click().run(timeout=40)
        self.assertEqual(app.exception, [])
        return app

    def test_access_portal_shows_company_sign_in_and_demo_bypass(self) -> None:
        app = AppTest.from_file(ROOT / "app.py", default_timeout=40).run()
        self.assertEqual(app.exception, [])
        self.assertTrue(any(button.label == "Open Northstar demo" for button in app.button))
        self.assertTrue(any("Open your company" in subheader.value for subheader in app.subheader))

    def test_populated_app_renders_and_local_assist_answers(self) -> None:
        with patch.dict(os.environ, {"SKILLVAULT_OLLAMA_URL": "http://127.0.0.1:9"}, clear=False):
            app = AppTest.from_file(ROOT / "app.py", default_timeout=40).run()
            app = self.open_northstar_demo(app)
            self.assertTrue(any("Chat with SkillVault" in subheader.value for subheader in app.subheader))
            app = app.chat_input[0].set_value(
                "How should I structure an investor presentation?"
            ).run(timeout=40)
            self.assertEqual(app.exception, [])
            self.assertEqual(len(app.chat_message), 2)

            app = app.sidebar.radio[0].set_value("Assist Mode").run(timeout=40)
            analyze = next(button for button in app.button if button.label == "Analyze case")
            app = analyze.click().run(timeout=40)
            self.assertEqual(app.exception, [])
            self.assertTrue(any(markdown.value == "### SkillVault answer" for markdown in app.markdown))
            self.assertTrue(any(markdown.value == "### Improve this recommendation" for markdown in app.markdown))

    def test_completed_work_mode_renders(self) -> None:
        with patch.dict(os.environ, {"SKILLVAULT_OLLAMA_URL": "http://127.0.0.1:9"}, clear=False):
            app = AppTest.from_file(ROOT / "app.py", default_timeout=40).run()
            app = self.open_northstar_demo(app)
            app = app.sidebar.radio[0].set_value("Add Company Knowledge").run(timeout=40)
            self.assertEqual(app.exception, [])
            self.assertTrue(any("Upload the work" in markdown.value for markdown in app.markdown))

    def test_visual_review_renders_with_offline_model(self) -> None:
        data = BytesIO()
        Image.new("RGB", (8, 8), "white").save(data, format="PNG")
        before = WorkFile("before.png", "image/png", data.getvalue())
        after = WorkFile("after.png", "image/png", data.getvalue())
        with patch.dict(os.environ, {"SKILLVAULT_OLLAMA_URL": "http://127.0.0.1:9"}, clear=False):
            app = self.open_northstar_demo(AppTest.from_file(ROOT / "app.py", default_timeout=40).run())
            app.session_state["completed_work_draft"] = capture_work([before], [after], [])
            app.session_state["capture_original_files"] = [before, after]
            app = app.sidebar.radio[0].set_value("Add Company Knowledge").run()
            self.assertEqual(app.exception, [])
            self.assertTrue(next(button for button in app.button if button.label == "Describe visible changes").disabled)

    def test_clean_start_copy_still_renders(self) -> None:
        app = AppTest.from_file(ROOT / "blank_app.py", default_timeout=40).run()
        self.assertEqual(app.exception, [])

    def test_discarding_one_proposal_moves_to_next_without_approval(self) -> None:
        with patch.dict(os.environ, {"SKILLVAULT_OLLAMA_URL": "http://127.0.0.1:9"}, clear=False):
            app = self.open_northstar_demo(AppTest.from_file(ROOT / "app.py", default_timeout=40).run())
            app.session_state["completed_work_draft"] = capture_work([], [], [], "Situation: First decision")
            app.session_state["capture_pending_drafts"] = [capture_work([], [], [], "Situation: Second decision")]
            app.session_state["capture_batch_split"] = True
            app = app.sidebar.radio[0].set_value("Add Company Knowledge").run()
            with patch("skillvault.storage.append_case") as save:
                app = next(button for button in app.button if button.label == "Discard this draft").click().run()
                save.assert_not_called()
            self.assertEqual(app.exception, [])
            self.assertEqual(app.session_state["completed_work_draft"].summary, "Second decision")
            self.assertEqual(app.session_state["capture_pending_drafts"], [])

    def test_capture_interview_updates_review_without_saving_knowledge(self) -> None:
        with patch.dict(os.environ, {"SKILLVAULT_OLLAMA_URL": "http://127.0.0.1:9"}, clear=False):
            app = self.open_northstar_demo(AppTest.from_file(ROOT / "app.py", default_timeout=40).run())
            app.session_state["completed_work_draft"] = capture_work([], [], [], "Situation: Investor deck revision")
            app = app.sidebar.radio[0].set_value("Add Company Knowledge").run()
            next(area for area in app.text_area if area.label.startswith("Why did you choose that approach?")).set_value("Investors asked for proof of demand")
            app = next(button for button in app.button if button.label == "Add answers to my decision").click().run()
            self.assertEqual(app.exception, [])
            self.assertEqual(app.session_state["completed_work_draft"].reasoning, "Investors asked for proof of demand")
            self.assertEqual(next(area for area in app.text_area if area.label == "Why the expert chose this approach").value, "Investors asked for proof of demand")


if __name__ == "__main__":
    unittest.main()
