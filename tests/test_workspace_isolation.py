from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from skillvault.workspace import (
    authenticate_user,
    find_workspace_record,
    register_company_workspace,
)


class WorkspaceIsolationTests(unittest.TestCase):
    def test_two_companies_receive_separate_storage_and_accounts(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            registry = root / "company_registry.json"
            workspaces = root / "company_workspaces"

            alpha_profile, _, alpha_paths = register_company_workspace(
                registry,
                workspaces,
                {"company_name": "Alpha Retail", "company_type": "Retail/e-commerce"},
                "Alpha Admin",
                "admin@alpha.example",
                "alpha-password",
            )
            beta_profile, _, beta_paths = register_company_workspace(
                registry,
                workspaces,
                {"company_name": "Beta Software", "company_type": "Software/SaaS"},
                "Beta Admin",
                "admin@beta.example",
                "beta-password",
            )

            self.assertNotEqual(alpha_profile["workspace_id"], beta_profile["workspace_id"])
            self.assertNotEqual(alpha_paths.root, beta_paths.root)
            self.assertNotEqual(alpha_paths.feedback, beta_paths.feedback)

            alpha_paths.cases.write_text('[{"company": "alpha"}]', encoding="utf-8")
            beta_paths.cases.write_text('[{"company": "beta"}]', encoding="utf-8")
            self.assertNotIn("beta", alpha_paths.cases.read_text(encoding="utf-8"))
            self.assertNotIn("alpha", beta_paths.cases.read_text(encoding="utf-8"))
            alpha_paths.feedback.write_text('[{"verdict": "helpful"}]', encoding="utf-8")
            self.assertFalse(beta_paths.feedback.exists())

            self.assertIsNotNone(
                authenticate_user(alpha_paths.users, "admin@alpha.example", "alpha-password")
            )
            self.assertIsNone(
                authenticate_user(alpha_paths.users, "admin@beta.example", "beta-password")
            )
            self.assertEqual(
                find_workspace_record(registry, str(beta_profile["workspace_id"]))["company_name"],
                "Beta Software",
            )


if __name__ == "__main__":
    unittest.main()
