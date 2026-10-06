"""Local company-workspace setup and sign-in helpers.

The local prototype keeps every company's profile, users, decisions, plans,
and retrieval index under a separate opaque workspace ID. Production hosting
can replace these JSON files with SSO and a tenant-scoped database while
preserving the same workspace boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import hmac
import json
import re
import secrets
from pathlib import Path
from threading import Lock


_REGISTRY_LOCK = Lock()


@dataclass(frozen=True)
class WorkspacePaths:
    """All persistent paths belonging to exactly one company workspace."""

    root: Path
    profile: Path
    users: Path
    cases: Path
    plans: Path
    feedback: Path
    index: Path


def _normalize_workspace_id(value: str) -> str:
    workspace_id = value.strip().lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{5,63}", workspace_id):
        raise ValueError("Enter a valid workspace ID.")
    return workspace_id


def _company_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    return (slug or "company")[:40].rstrip("-")


def workspace_paths(workspaces_root: Path, workspace_id: str) -> WorkspacePaths:
    """Return tenant-scoped paths while rejecting path traversal."""

    normalized = _normalize_workspace_id(workspace_id)
    resolved_root = workspaces_root.resolve()
    workspace_root = (resolved_root / normalized).resolve()
    if not workspace_root.is_relative_to(resolved_root):
        raise ValueError("Workspace path is outside the configured company-data directory.")
    return WorkspacePaths(
        root=workspace_root,
        profile=workspace_root / "workspace.json",
        users=workspace_root / "users.json",
        cases=workspace_root / "approved_expert_decisions.json",
        plans=workspace_root / "approved_plans.json",
        feedback=workspace_root / "recommendation_feedback.json",
        index=workspace_root / "knowledge_index",
    )


def _read_json(path: Path, default: object) -> object:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_workspace(path: Path) -> dict[str, object]:
    payload = _read_json(path, {})
    return payload if isinstance(payload, dict) else {}


def save_workspace(path: Path, profile: dict[str, object]) -> None:
    _write_json(path, profile)


def _password_hash(password: str, salt_hex: str) -> str:
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), 210_000)
    return digest.hex()


def create_user(
    path: Path,
    display_name: str,
    email: str,
    password: str,
    *,
    role: str = "administrator",
) -> dict[str, str]:
    users = _read_json(path, [])
    if not isinstance(users, list):
        users = []
    normalized = email.strip().lower()
    if not display_name.strip():
        raise ValueError("Enter the employee's name.")
    if "@" not in normalized:
        raise ValueError("Enter a valid work email.")
    if len(password) < 8:
        raise ValueError("Use a password with at least 8 characters.")
    if any(isinstance(user, dict) and user.get("email") == normalized for user in users):
        raise ValueError("An account with that email already exists.")
    salt = secrets.token_hex(16)
    user = {
        "user_id": secrets.token_hex(12),
        "display_name": display_name.strip(),
        "email": normalized,
        "role": role.strip().lower() or "employee",
        "salt": salt,
        "password_hash": _password_hash(password, salt),
    }
    users.append(user)
    _write_json(path, users)
    return {
        "user_id": str(user["user_id"]),
        "display_name": str(user["display_name"]),
        "email": str(user["email"]),
        "role": str(user["role"]),
    }


def authenticate_user(path: Path, email: str, password: str) -> dict[str, str] | None:
    users = _read_json(path, [])
    if not isinstance(users, list):
        return None
    normalized = email.strip().lower()
    for user in users:
        if not isinstance(user, dict) or user.get("email") != normalized:
            continue
        salt = str(user.get("salt", ""))
        expected = str(user.get("password_hash", ""))
        if salt and expected and hmac.compare_digest(_password_hash(password, salt), expected):
            return {
                "user_id": str(user.get("user_id", "")),
                "display_name": str(user.get("display_name", "User")),
                "email": normalized,
                "role": str(user.get("role", "employee")),
            }
    return None


def list_workspace_records(registry_path: Path) -> list[dict[str, str]]:
    """Load the private registry used to resolve an entered workspace ID."""

    records = _read_json(registry_path, [])
    if not isinstance(records, list):
        return []
    return [
        {str(key): str(value) for key, value in record.items()}
        for record in records
        if isinstance(record, dict)
    ]


def find_workspace_record(registry_path: Path, workspace_id: str) -> dict[str, str] | None:
    """Resolve one company without exposing the names of other companies."""

    try:
        normalized = _normalize_workspace_id(workspace_id)
    except ValueError:
        return None
    return next(
        (
            record
            for record in list_workspace_records(registry_path)
            if record.get("workspace_id") == normalized
        ),
        None,
    )


def register_company_workspace(
    registry_path: Path,
    workspaces_root: Path,
    profile: dict[str, object],
    admin_name: str,
    admin_email: str,
    password: str,
) -> tuple[dict[str, object], dict[str, str], WorkspacePaths]:
    """Create an isolated company workspace and its first administrator."""

    company_name = str(profile.get("company_name", "")).strip()
    if not company_name:
        raise ValueError("Enter a company name.")

    with _REGISTRY_LOCK:
        existing = list_workspace_records(registry_path)
        existing_ids = {record.get("workspace_id", "") for record in existing}
        while True:
            workspace_id = f"{_company_slug(company_name)}-{secrets.token_hex(3)}"
            if workspace_id not in existing_ids:
                break

        paths = workspace_paths(workspaces_root, workspace_id)
        company_profile = dict(profile)
        company_profile.update(
            {
                "workspace_id": workspace_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        administrator = create_user(
            paths.users,
            admin_name,
            admin_email,
            password,
            role="administrator",
        )
        save_workspace(paths.profile, company_profile)
        existing.append(
            {
                "workspace_id": workspace_id,
                "company_name": company_name,
                "created_at": str(company_profile["created_at"]),
            }
        )
        _write_json(registry_path, existing)
    return company_profile, administrator, paths
