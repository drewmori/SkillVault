from __future__ import annotations

from datetime import date, timedelta
from html import escape
from hashlib import sha256
import os
from pathlib import Path

import streamlit as st
import pandas as pd

from skillvault.data import (
    ExpertCase,
    LIFECYCLE_STATUSES,
    VALID_DECISIONS,
    build_demo_cases,
    cases_from_dataframe,
    decision_is_retrievable,
    effective_lifecycle_status,
    infer_features,
)
from skillvault.engine import SkillVaultEngine
from skillvault.capture import capture_work, next_interview_question, apply_interview
from skillvault.current_case import prepare_current_case, material_followup
from skillvault.artifacts import store_artifacts, read_artifact
from skillvault.semantic_search import SemanticIndex, SemanticSearchError
from skillvault.speech import transcribe_recording, TranscriptionError
from skillvault.feedback_learning import learning_profile, learning_instructions
from skillvault.importer import WorkFile, draft_from_completed_work
from skillvault.local_model import LocalModelError, SkillVaultLocalModel
from skillvault.storage import (
    append_case,
    load_approved_plans,
    load_cases,
    load_recommendation_feedback,
    save_approved_plan,
    save_cases,
    save_plan_outcome,
    save_recommendation_feedback,
)
from skillvault.workspace import (
    authenticate_user,
    find_workspace_record,
    load_workspace,
    register_company_workspace,
    workspace_paths,
)


ROOT = Path(__file__).parent
COMPANY_REGISTRY_PATH = ROOT / "data" / "local" / "company_registry.json"
COMPANY_WORKSPACES_ROOT = ROOT / "data" / "local" / "company_workspaces"
SCENARIO_OPTIONS = [
    "All scenarios",
    "Client support",
    "Engineering/code",
    "Presentations & pitches",
    "Reporting & analytics",
    "Security & access",
    "Billing & operations",
    "Onboarding & documentation",
    "Other",
]


st.set_page_config(
    page_title="SkillVault",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


def app_setting(name: str, default: str = "") -> str:
    environment_value = os.getenv(name, "").strip()
    if environment_value:
        return environment_value
    try:
        return str(st.secrets.get(name, default)).strip()
    except Exception:
        return default


def setting_is_enabled(name: str, default: bool = False) -> bool:
    fallback = "true" if default else "false"
    return app_setting(name, fallback).lower() in {"1", "true", "yes", "on"}


def enter_workspace(profile: dict[str, object], user: dict[str, str]) -> None:
    """Store only the authenticated company identity in this browser session."""

    st.session_state["skillvault_context"] = {
        "mode": "company",
        "workspace_id": str(profile["workspace_id"]),
        "company_name": str(profile["company_name"]),
        "user": user,
    }


def leave_workspace() -> None:
    """Remove user-specific state before returning to the access portal."""

    st.session_state.clear()
    st.rerun()


def render_access_portal() -> None:
    """Render company sign-in/setup and the explicit development bypass."""

    st.markdown(
        """
        <style>
        .stApp { background: radial-gradient(circle at 82% 0%, rgba(112,72,232,.12), transparent 30rem), #f6f8fc; }
        .block-container { max-width: 1080px; padding-top: 2.2rem; padding-bottom: 4rem; }
        .access-hero { padding: 2.2rem 2.35rem; border-radius: 26px; color: white; background: linear-gradient(120deg,#101a32 0%,#263f7e 58%,#6844bd 100%); box-shadow: 0 22px 55px rgba(31,48,100,.20); margin-bottom: 1.4rem; }
        .access-hero h1 { margin: 0; font-size: 2.55rem; letter-spacing: -.04em; }
        .access-hero p { color: #dce7ff; margin: .55rem 0 0; font-size: 1.08rem; max-width: 760px; }
        .access-kicker { color: #c7d3ff; font-size: .72rem; font-weight: 800; letter-spacing: .14em; text-transform: uppercase; margin-bottom: .65rem; }
        div[data-testid="stTextArea"] textarea, div[data-testid="stTextInput"] input { border-radius: 12px; background: white; }
        div[data-testid="stFormSubmitButton"] > button, div[data-testid="stButton"] > button { border-radius: 11px; font-weight: 700; }
        </style>
        <div class="access-hero">
          <div class="access-kicker">Private company knowledge</div>
          <h1>SkillVault</h1>
          <p>Sign in to your company workspace. Retrieval, uploads, feedback, plans, and search indexes stay scoped to that company.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sign_in_tab, create_tab = st.tabs(["Sign in", "Create company workspace"])
    with sign_in_tab:
        st.subheader("Open your company’s SkillVault")
        st.caption("Your administrator provides the workspace ID. Company names are not publicly listed.")
        with st.form("company_sign_in_form"):
            workspace_id = st.text_input(
                "Workspace ID",
                placeholder="example-company-a1b2c3",
                help="This identifies which private company database and search index to use.",
            )
            email = st.text_input("Work email")
            password = st.text_input("Password", type="password")
            sign_in_submitted = st.form_submit_button(
                "Sign in to company workspace",
                type="primary",
                width="stretch",
            )
        if sign_in_submitted:
            record = find_workspace_record(COMPANY_REGISTRY_PATH, workspace_id)
            if record is None:
                st.error("The workspace ID, email, or password is incorrect.")
            else:
                paths = workspace_paths(COMPANY_WORKSPACES_ROOT, record["workspace_id"])
                profile = load_workspace(paths.profile)
                user = authenticate_user(paths.users, email, password)
                if not profile or user is None:
                    st.error("The workspace ID, email, or password is incorrect.")
                else:
                    enter_workspace(profile, user)
                    st.rerun()

    with create_tab:
        st.subheader("Create an isolated company workspace")
        st.caption("This local prototype creates a separate folder, account list, knowledge file, plan file, and retrieval index for the company.")
        with st.form("company_workspace_setup_form"):
            company_col, type_col = st.columns(2)
            with company_col:
                company_name = st.text_input("Company name")
            with type_col:
                company_type = st.selectbox(
                    "Company type",
                    [
                        "Software/SaaS",
                        "Manufacturing",
                        "Healthcare",
                        "Financial services",
                        "Retail/e-commerce",
                        "Professional services",
                        "Education",
                        "Nonprofit",
                        "Government",
                        "Other",
                    ],
                )
            industry_col, size_col = st.columns(2)
            with industry_col:
                industry = st.text_input("Industry or specialty", placeholder="Example: industrial robotics maintenance")
            with size_col:
                company_size = st.selectbox("Company size", ["1–10", "11–50", "51–200", "201–1,000", "1,001–5,000", "5,000+"])
            primary_team = st.text_input("Primary team using SkillVault", placeholder="Example: customer support and engineering")
            description = st.text_area(
                "What does the company do?",
                placeholder="Describe its customers, products or services, and the work employees need help with.",
                height=95,
            )
            knowledge_goal = st.text_area(
                "What knowledge should SkillVault preserve?",
                placeholder="Example: troubleshooting decisions, escalation rules, client communication, reporting, and technical methods.",
                height=95,
            )
            sensitivity = st.selectbox(
                "Expected data sensitivity",
                [
                    "Internal operating knowledge",
                    "May include confidential client information",
                    "May include regulated or highly sensitive information",
                ],
            )
            st.markdown("#### First workspace administrator")
            name_col, email_col = st.columns(2)
            with name_col:
                admin_name = st.text_input("Administrator name")
            with email_col:
                admin_email = st.text_input("Administrator work email")
            password_col, confirm_col = st.columns(2)
            with password_col:
                new_password = st.text_input("Create password", type="password", help="Use at least 8 characters.")
            with confirm_col:
                confirm_password = st.text_input("Confirm password", type="password")
            create_submitted = st.form_submit_button(
                "Create private workspace",
                type="primary",
                width="stretch",
            )

        if create_submitted:
            required = {
                "company name": company_name,
                "industry or specialty": industry,
                "primary team": primary_team,
                "company description": description,
                "knowledge goal": knowledge_goal,
                "administrator name": admin_name,
                "administrator email": admin_email,
            }
            missing = [name for name, value in required.items() if not value.strip()]
            if missing:
                st.error(f"Please complete: {', '.join(missing)}")
            elif new_password != confirm_password:
                st.error("The passwords do not match.")
            else:
                profile = {
                    "company_name": company_name.strip(),
                    "company_type": company_type,
                    "industry": industry.strip(),
                    "company_size": company_size,
                    "primary_team": primary_team.strip(),
                    "description": description.strip(),
                    "knowledge_goal": knowledge_goal.strip(),
                    "sensitivity": sensitivity,
                }
                try:
                    saved_profile, administrator, _ = register_company_workspace(
                        COMPANY_REGISTRY_PATH,
                        COMPANY_WORKSPACES_ROOT,
                        profile,
                        admin_name,
                        admin_email,
                        new_password,
                    )
                except (OSError, ValueError) as error:
                    st.error(str(error))
                else:
                    enter_workspace(saved_profile, administrator)
                    st.session_state["workspace_created"] = True
                    st.rerun()

    if setting_is_enabled("SKILLVAULT_ENABLE_DEMO_BYPASS", True):
        st.divider()
        with st.container(border=True):
            st.markdown("#### Development bypass")
            st.write("Open the original populated Northstar Cloud competition demo without creating or signing into a company workspace.")
            st.warning("Development/demo access only. Set `SKILLVAULT_ENABLE_DEMO_BYPASS=false` before a real company deployment.")
            if st.button(
                "Open Northstar demo",
                icon=":material/developer_mode:",
                width="stretch",
            ):
                st.session_state["skillvault_context"] = {
                    "mode": "northstar_demo",
                    "workspace_id": "northstar-demo",
                    "company_name": "Northstar Cloud",
                    "user": {
                        "display_name": "Development reviewer",
                        "email": "demo@northstar.example",
                        "role": "demo",
                    },
                }
                st.rerun()

    st.caption("Prototype authentication uses salted password hashes. Production should use company SSO, encrypted storage, audit logs, and role-based access controls.")


def require_workspace_context() -> dict[str, object]:
    context = st.session_state.get("skillvault_context")
    if isinstance(context, dict) and context.get("mode") == "northstar_demo":
        if setting_is_enabled("SKILLVAULT_ENABLE_DEMO_BYPASS", True):
            return context
        st.session_state.pop("skillvault_context", None)
    elif isinstance(context, dict) and context.get("mode") == "company":
        workspace_id = str(context.get("workspace_id", ""))
        record = find_workspace_record(COMPANY_REGISTRY_PATH, workspace_id)
        if record is not None:
            paths = workspace_paths(COMPANY_WORKSPACES_ROOT, workspace_id)
            if paths.profile.exists() and paths.users.exists():
                return context
        st.session_state.pop("skillvault_context", None)
        st.error("That company workspace is no longer available. Sign in again.")
    render_access_portal()
    st.stop()
    raise RuntimeError("Streamlit stopped before a workspace was selected.")


workspace_context = require_workspace_context()
IS_NORTHSTAR_DEMO = workspace_context["mode"] == "northstar_demo"
WORKSPACE_ID = str(workspace_context["workspace_id"])
CURRENT_USER = dict(workspace_context.get("user", {}))

if IS_NORTHSTAR_DEMO:
    COMPANY_PROFILE: dict[str, object] = {
        "company_name": "Northstar Cloud",
        "company_type": "Software/SaaS",
        "industry": "Cloud software and technical support",
        "primary_team": "Support and engineering",
    }
    APPROVED_CASES_PATH = ROOT / "approved_expert_decisions.json"
    APPROVED_PLANS_PATH = ROOT / "approved_plans.json"
    RECOMMENDATION_FEEDBACK_PATH = ROOT / "recommendation_feedback.json"
    KNOWLEDGE_INDEX_PATH = ROOT / "data" / "local" / "indexes" / "northstar_knowledge"
else:
    company_paths = workspace_paths(COMPANY_WORKSPACES_ROOT, WORKSPACE_ID)
    COMPANY_PROFILE = load_workspace(company_paths.profile)
    APPROVED_CASES_PATH = company_paths.cases
    APPROVED_PLANS_PATH = company_paths.plans
    RECOMMENDATION_FEEDBACK_PATH = company_paths.feedback
    KNOWLEDGE_INDEX_PATH = company_paths.index

COMPANY_NAME = str(COMPANY_PROFILE.get("company_name", "Your company"))
ARTIFACTS_PATH = (
    ROOT / "data" / "local" / "northstar_artifacts"
    if IS_NORTHSTAR_DEMO else company_paths.root / "artifacts"
)


def current_company_cases() -> list[ExpertCase]:
    stored_cases = load_cases(APPROVED_CASES_PATH)
    replaced_sources = {
        case.replaces_source.strip().lower()
        for case in stored_cases
        if case.replaces_source.strip() and decision_is_retrievable(case)
    }
    built_in_cases = build_demo_cases() if IS_NORTHSTAR_DEMO else []
    return [
        case
        for case in built_in_cases + stored_cases
        if decision_is_retrievable(case) and case.source_file.strip().lower() not in replaced_sources
    ]


def _case_file_version(path: Path) -> tuple[int, int]:
    if not path.exists():
        return (0, 0)
    stat = path.stat()
    return (stat.st_mtime_ns, stat.st_size)


@st.cache_resource(max_entries=32)
def get_engine(
    workspace_id: str,
    case_file: str,
    index_file: str,
    version: tuple[int, int],
    include_demo_cases: bool,
    embedding_model: str = "embeddinggemma",
    embedding_url: str = "http://127.0.0.1:11434",
    semantic_version: tuple[int, int] = (0, 0),
) -> SkillVaultEngine:
    # Every tenant identifier and path is part of the cache key. Never return
    # another company's engine just because both sessions share a server.
    del workspace_id, version
    stored_cases = load_cases(Path(case_file))
    replaced_sources = {
        case.replaces_source.strip().lower()
        for case in stored_cases
        if case.replaces_source.strip() and decision_is_retrievable(case)
    }
    built_in_cases = build_demo_cases() if include_demo_cases else []
    active_cases = [
        case
        for case in built_in_cases + stored_cases
        if decision_is_retrievable(case) and case.source_file.strip().lower() not in replaced_sources
    ]
    engine = SkillVaultEngine(
        active_cases,
        index_path=Path(index_file),
        semantic_index=SemanticIndex(Path(index_file).with_suffix(".semantic.sqlite3"), embedding_model, embedding_url),
    )
    engine.train()
    return engine


@st.cache_resource
def get_local_model(model: str, base_url: str) -> SkillVaultLocalModel:
    return SkillVaultLocalModel(model=model, base_url=base_url)


engine = get_engine(
    WORKSPACE_ID,
    str(APPROVED_CASES_PATH),
    str(KNOWLEDGE_INDEX_PATH),
    _case_file_version(APPROVED_CASES_PATH),
    IS_NORTHSTAR_DEMO,
    app_setting("SKILLVAULT_EMBEDDING_MODEL", "embeddinggemma"),
    app_setting("SKILLVAULT_OLLAMA_URL", "http://127.0.0.1:11434"),
    _case_file_version(KNOWLEDGE_INDEX_PATH.with_suffix(".semantic.sqlite3")),
)
local_model = get_local_model(
    app_setting("SKILLVAULT_LOCAL_MODEL", "qwen3.5:9b"),
    app_setting("SKILLVAULT_OLLAMA_URL", "http://127.0.0.1:11434"),
)


def parse_iso_date(value: object, fallback: date) -> date:
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return fallback


def contextual_chat_query(prompt: str, messages: list[dict[str, object]]) -> str:
    """Add the previous user turn only when a new chat turn looks like a follow-up."""
    lowered = prompt.lower().strip()
    follow_up_signals = (
        "what about",
        "how about",
        "that ",
        "this ",
        "it ",
        "also",
        "instead",
        "same",
        "then",
        "more detail",
    )
    looks_like_follow_up = len(prompt.split()) < 18 or any(signal in lowered for signal in follow_up_signals)
    if not looks_like_follow_up:
        return prompt
    previous_questions = [
        str(message.get("content", "")).strip()
        for message in messages
        if message.get("role") == "user" and str(message.get("content", "")).strip()
    ]
    if not previous_questions:
        return prompt
    context = "\n".join(question[:1200] for question in previous_questions[-3:])
    return f"Recent employee context:\n{context}\nCurrent follow-up question: {prompt}"


def render_decision_lineage(lineage: dict[str, object]) -> None:
    """Render the auditable path from the current question to its decision status."""
    retrieval = lineage.get("retrieval", {})
    retrieved = lineage.get("retrieved_decisions", [])
    kept = lineage.get("judgment_kept", [])
    changes = lineage.get("changes_for_today", [])
    missing = lineage.get("missing_context", [])
    final_action = lineage.get("final_action", {})
    outcome = lineage.get("outcome", {})

    st.markdown("### Decision lineage")
    st.caption("See exactly how SkillVault moved from the employee’s question to an evidence-linked current plan.")
    st.markdown("**Current problem → Retrieved evidence → Expert judgment kept → Current-case changes → Approval → Outcome**")

    metric_one, metric_two, metric_three = st.columns(3)
    metric_one.metric("Knowledge match", str(retrieval.get("match_strength", "Unknown")))
    metric_two.metric("Decisions retrieved", len(retrieved))
    metric_three.metric("Decision status", str(final_action.get("approval_status", "Awaiting approval")))
    st.caption(
        f"Raw similarity {float(retrieval.get('similarity', 0.0)):.0%} · "
        f"{int(retrieval.get('candidate_count', 0)):,} candidates searched from "
        f"{int(retrieval.get('indexed_count', 0)):,} indexed decisions"
    )

    with st.container(border=True):
        st.markdown("#### 1. Current problem")
        st.write(str(lineage.get("current_problem", "")))

    with st.container(border=True):
        st.markdown("#### 2. Retrieved past decisions")
        for decision in retrieved:
            st.write(
                f"**{float(decision.get('similarity', 0.0)):.0%} match · "
                f"{decision.get('expert_approach', 'Unknown approach')}** — "
                f"{decision.get('summary', '')}"
            )
            if decision.get("goal"):
                st.write(f"**Goal:** {decision['goal']}")
            if decision.get("chosen_approach"):
                st.write(f"**Choice made:** {decision['chosen_approach']}")
            if decision.get("reusable_rule"):
                st.write(f"**Reusable rule:** {decision['reusable_rule']}")
            st.caption(
                f"Source: {decision.get('source_file', 'No source recorded')} · "
                f"Owner: {decision.get('expert_owner', 'Unknown')} · "
                f"Department: {decision.get('department', 'Unassigned')} · "
                f"Reviewed: {decision.get('last_reviewed_date', 'Not recorded')} · "
                f"Expires: {decision.get('expiration_date', 'Not recorded')}"
            )

    with st.container(border=True):
        st.markdown("#### 3. Expert judgment carried forward")
        for item in kept:
            st.write(f"- {item.get('action', '')}")
            method = str(item.get("method", "")).strip()
            method_note = f" · Method: {method}" if method else ""
            st.caption(f"Source: {item.get('source_file', 'No source recorded')}{method_note}")

    with st.container(border=True):
        st.markdown("#### 4. What SkillVault changed for today")
        for change in changes:
            st.write(f"- {change}")
        st.markdown("**Missing context still to confirm**")
        if missing:
            for question in missing:
                st.write(f"- {question}")
        else:
            st.write("- No obvious core detail is missing; current owners, policies, and success checks still require confirmation.")

    with st.container(border=True):
        st.markdown("#### 5. Final action, approval, and outcome")
        status = str(final_action.get("approval_status", "Awaiting approval"))
        if status == "Approved":
            st.success("This adapted plan was approved and saved with its evidence lineage.")
        elif status == "Expert review requested":
            st.warning("This plan is awaiting a decision from the responsible expert.")
        else:
            st.info("This adapted plan has not been approved yet.")
        st.write(f"**Recommended approach:** {final_action.get('recommended_approach', '')}")
        with st.expander("View the adapted final action"):
            st.markdown(str(final_action.get("adapted_plan", "")))
        st.write(f"**Outcome:** {outcome.get('status', 'Pending execution')}")
        if outcome.get("summary"):
            st.write(str(outcome.get("summary")))
        if outcome.get("effectiveness"):
            st.caption(f"Effectiveness: {outcome.get('effectiveness')}")
        if lineage.get("plan_id"):
            st.caption(f"Saved plan ID: {lineage['plan_id']}")

st.markdown(
    """
    <style>
    :root { --ink: #172033; --muted: #667085; --line: #e3e8f1; --blue: #3156d9; --violet: #7048e8; --mint: #1fa774; }
    .stApp { background: radial-gradient(circle at 82% 0%, rgba(112,72,232,.08), transparent 28rem), #f7f9fc; color: var(--ink); }
    .block-container {max-width: 1240px; padding-top: 1.25rem; padding-bottom: 4rem;}
    [data-testid="stSidebar"] { background: #101727; border-right: 1px solid #202b43; }
    [data-testid="stSidebar"] * { color: #dbe5ff !important; }
    [data-testid="stSidebar"] .stRadio label { padding: .5rem .65rem; border-radius: 10px; }
    [data-testid="stSidebar"] .stRadio label:hover { background: #1a2744; }
    [data-testid="stSidebar"] hr { border-color: #2b3855; }
    .hero { position: relative; overflow: hidden; padding: 1.8rem 2rem; border-radius: 24px; background: linear-gradient(120deg, #111a31 0%, #233a72 55%, #5d3eae 100%); color: white; margin-bottom: 1.5rem; box-shadow: 0 18px 45px rgba(31,48,100,.18); }
    .hero:after { content: ""; position: absolute; width: 270px; height: 270px; right: -55px; top: -115px; border-radius: 50%; border: 1px solid rgba(255,255,255,.22); box-shadow: 0 0 0 22px rgba(255,255,255,.04), 0 0 0 44px rgba(255,255,255,.03); }
    .hero h1 { margin: 0; font-size: 2.45rem; letter-spacing: -.04em; position: relative; z-index: 1; }
    .hero p { margin: .45rem 0 0; color: #d9e8ff; font-size: 1.08rem; position: relative; z-index: 1; }
    .hero-kicker { display: inline-block; color: #b8c8ff; font-size: .72rem; font-weight: 800; letter-spacing: .13em; text-transform: uppercase; margin-bottom: .65rem; position: relative; z-index: 1; }
    .section-kicker { color: #6d5bd0; font-size: .72rem; font-weight: 800; letter-spacing: .12em; text-transform: uppercase; margin-top: .3rem; }
    .metric-card { padding: 1rem 1.1rem; border: 1px solid var(--line); border-radius: 16px; background: rgba(255,255,255,.82); box-shadow: 0 8px 24px rgba(32,48,86,.05); }
    .evidence-card { padding: .85rem 1rem; border: 1px solid #dfe5f2; border-left: 4px solid var(--blue); border-radius: 14px; background: white; margin-bottom: .65rem; box-shadow: 0 6px 18px rgba(32,48,86,.04); }
    .confidence-high { color: var(--mint); font-weight: 800; }
    .confidence-mid { color: #b36b00; font-weight: 800; }
    div[data-testid="stTextArea"] textarea, div[data-testid="stTextInput"] input { border-radius: 12px; border: 1px solid #d4dceb; background: white; }
    div[data-testid="stSelectbox"] label { color: var(--ink) !important; }
    div[data-testid="stSelectbox"] [data-baseweb="select"], div[data-testid="stSelectbox"] [data-baseweb="select"] * { color: var(--ink) !important; background: white !important; }
    div[data-testid="stButton"] > button { border-radius: 11px; border: 1px solid #cfd8ee; font-weight: 700; transition: all .15s ease; }
    div[data-testid="stButton"] > button:hover { border-color: var(--blue); color: var(--blue); transform: translateY(-1px); box-shadow: 0 6px 15px rgba(49,86,217,.12); }
    div[data-testid="stFormSubmitButton"] > button { border-radius: 11px; font-weight: 750; }
    div[data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 13px; background: white; }
    .footer-note { color: #7a8499; font-size: .78rem; margin-top: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

company_name_html = escape(COMPANY_NAME.upper())
company_type_html = escape(str(COMPANY_PROFILE.get("company_type", "company")))
hero_kicker = (
    "NORTHSTAR CLOUD · DEVELOPMENT DEMO"
    if IS_NORTHSTAR_DEMO
    else f"{company_name_html} · PRIVATE KNOWLEDGE WORKSPACE"
)
hero_description = (
    "An AI apprentice that preserves expert judgment for SaaS teams."
    if IS_NORTHSTAR_DEMO
    else f"An isolated SkillVault workspace for {company_type_html} teams."
)
st.markdown(
    f'<div class="hero"><div class="hero-kicker">{hero_kicker}</div><h1>🧠 SkillVault</h1><p>{hero_description}</p></div>',
    unsafe_allow_html=True,
)

if st.session_state.pop("workspace_created", False):
    st.success(
        f"{COMPANY_NAME} was created with an empty private knowledge base. Save this workspace ID: `{WORKSPACE_ID}`"
    )

with st.sidebar:
    st.markdown("### ✦ SkillVault")
    st.caption(f"{COMPANY_NAME} / expert memory")
    if IS_NORTHSTAR_DEMO:
        st.caption("Development bypass · populated fictional data")
    else:
        st.caption(f"Workspace ID: `{WORKSPACE_ID}`")
        st.caption(
            f"Signed in as {CURRENT_USER.get('display_name', 'Employee')} · "
            f"{CURRENT_USER.get('role', 'employee')}"
        )
    all_modes = [
        "SkillVault Chat",
        "Assist Mode",
        "Add Company Knowledge",
        "Add Expert Decision",
        "Knowledge Governance",
        "Outcome Review",
        "Model Insights",
    ]
    available_modes = (
        all_modes
        if engine.cases
        else [
            "Add Company Knowledge",
            "Add Expert Decision",
            "Knowledge Governance",
            "Outcome Review",
            "Model Insights",
        ]
    )
    mode = st.radio(
        "Choose a mode",
        available_modes,
    )
    st.caption(
        "Demo company: Northstar Cloud SaaS support and engineering"
        if IS_NORTHSTAR_DEMO
        else f"Private company scope · {len(engine.cases):,} approved decisions"
    )
    local_model_status = local_model.status()
    if local_model_status.available:
        st.caption(f"Answer engine: local {local_model.model} · no external LLM API")
    elif local_model_status.reachable:
        st.caption(f"Local model setup needed: pull {local_model.model}")
    else:
        st.caption("Answer engine: local fallback · Ollama not running")
    st.divider()
    if st.button(
        "Exit Northstar demo" if IS_NORTHSTAR_DEMO else "Sign out",
        icon=":material/logout:",
        width="stretch",
    ):
        leave_workspace()

knowledge_method = None
if mode == "Add Company Knowledge":
    st.markdown('<div class="section-kicker">Knowledge capture</div>', unsafe_allow_html=True)
    st.subheader("Add company knowledge")
    st.write("Add evidence from completed work or import decisions that are already organized in a spreadsheet.")
    knowledge_method = st.segmented_control(
        "How do you want to add knowledge?",
        ["Upload completed work", "Import prepared CSV"],
        default="Upload completed work",
        key="knowledge_import_method",
    )

if mode == "SkillVault Chat":
    st.markdown('<div class="section-kicker">Private company copilot</div>', unsafe_allow_html=True)
    st.subheader("Chat with SkillVault")
    st.write(
        f"Ask follow-up questions naturally. SkillVault retrieves approved {COMPANY_NAME} decisions for each turn, "
        "then a local open-weight model rewrites that evidence for the current situation."
    )

    chat_messages = st.session_state.setdefault("skillvault_chat_messages", [])
    control_col, clear_col = st.columns([3, 1])
    with control_col:
        chat_focus = st.selectbox(
            "Knowledge focus area",
            SCENARIO_OPTIONS,
            key="skillvault_chat_focus",
            help="Prioritize approved decisions from one work area.",
        )
    with clear_col:
        st.write("")
        if st.button("Clear chat", icon=":material/delete_sweep:", width="stretch"):
            st.session_state["skillvault_chat_messages"] = []
            st.rerun()

    st.session_state.setdefault("employee_team", "")
    chat_team = st.text_input(
        "Your team (optional)", value=st.session_state["employee_team"], key="chat_employee_team",
        help="Used to tailor the answer, not treated as proof of company policy.",
    )
    st.session_state["employee_team"] = chat_team
    chat_context_file = st.file_uploader(
        "Current ticket or document (optional)",
        type=["txt", "md", "log", "csv", "json", "pdf", "docx", "pptx", "py", "js", "ts", "sql", "eml"],
        key="chat_current_file",
        help="Used only as context for your questions in this chat. It is not added to approved company knowledge.",
    )
    if chat_context_file:
        st.caption("This file will inform each new chat question until you remove it. It is not added to approved decisions; answers may summarize it.")

    if local_model_status.available:
        st.success(
            f"Local model connected: {local_model.model}. Prompts and retrieved decisions stay inside the configured private Ollama deployment.",
            icon=":material/memory:",
        )
    else:
        st.warning(
            "The local language model is not ready, so chat will use the simpler evidence-grounded fallback until Ollama is installed and running."
        )
        with st.expander("Local model setup"):
            st.code("winget install Ollama.Ollama\nollama pull qwen3.5:9b", language="powershell")
            st.caption(
                "After installation, restart the terminal and Streamlit. The 9B model download is about 6.6 GB. "
                "Use qwen3.5:4b on a lower-memory computer."
            )

    if IS_NORTHSTAR_DEMO:
        suggestions = {
            "Client upload failure": "A client's uploads fail above 100 MB but smaller files work. What should I do and tell them?",
            "Investor presentation": f"How should I structure an investor presentation using {COMPANY_NAME}'s past decisions?",
            "Production code issue": "A recent API change caused duplicate webhook events. How should I investigate and communicate it?",
        }
    else:
        suggestions = {
            f"Company example {index}": f"I have a new situation related to this past decision: {case.summary} What should I do now?"
            for index, case in enumerate(engine.cases[:3], start=1)
        }
    suggested_prompt = ""
    if not chat_messages:
        selected_suggestion = st.pills(
            "Try asking",
            list(suggestions),
            label_visibility="collapsed",
        )
        if selected_suggestion:
            suggested_prompt = suggestions[selected_suggestion]

    for message in chat_messages:
        with st.chat_message(str(message.get("role", "assistant"))):
            st.markdown(str(message.get("content", "")))
            if message.get("role") == "assistant":
                st.caption(
                    f"{message.get('engine', 'SkillVault')} · "
                    f"Knowledge match: {message.get('match', 'Unknown')}"
                )
                sources = message.get("sources", [])
                if sources:
                    with st.expander("Sources used"):
                        for source in sources:
                            st.write(f"- `{source}`")

    typed_prompt = st.chat_input(
        "Ask about a client, presentation, report, code issue, or company process",
        submit_mode="disable",
    )
    ask_about_file = st.button(
        "Ask about attached work", help="Start with the attached ticket or document; no written question is required.",
    ) if chat_context_file else False
    incoming_prompt = suggested_prompt or typed_prompt or ("What should I do with this current work?" if ask_about_file else "")
    if incoming_prompt:
        current_case = prepare_current_case(
            incoming_prompt, chat_team,
            WorkFile(chat_context_file.name, chat_context_file.type or "application/octet-stream", chat_context_file.getvalue()) if chat_context_file else None,
        )
        feedback_profile = learning_profile(load_recommendation_feedback(RECOMMENDATION_FEEDBACK_PATH), incoming_prompt, WORKSPACE_ID)
        prior_messages = list(chat_messages)
        retrieval_query = contextual_chat_query(current_case.retrieval_question(), prior_messages)
        result = engine.predict(retrieval_query, chat_focus)
        chat_messages.append({"role": "user", "content": incoming_prompt})
        with st.chat_message("user"):
            st.markdown(incoming_prompt)

        generation_error = ""
        with st.chat_message("assistant"):
            if current_case.warning:
                st.warning(current_case.warning)
            if local_model.available:
                try:
                    with st.spinner(f"Local {local_model.model} is adapting approved decisions..."):
                        answer = local_model.grounded_answer(
                            current_case.model_question(),
                            result,
                            conversation_history=[
                                {
                                    "role": str(item.get("role", "")),
                                    "content": str(item.get("content", "")),
                                }
                                for item in prior_messages
                            ],
                            company_profile=COMPANY_PROFILE,
                            feedback_profile=feedback_profile,
                        )
                    answer_engine = f"Local {local_model.model}"
                except LocalModelError as error:
                    answer = engine.compose_response(current_case.model_question(), result, feedback_profile=feedback_profile)
                    answer_engine = "Local evidence-grounded fallback"
                    generation_error = str(error)
            else:
                answer = engine.compose_response(current_case.model_question(), result, feedback_profile=feedback_profile)
                answer_engine = "Local evidence-grounded fallback"
            if generation_error:
                st.warning(f"The local model could not answer, so SkillVault used its fallback: {generation_error}")
            followup = material_followup(current_case, result.knowledge_match)
            if followup:
                answer += f"\n\n**One detail that could change this plan:** {followup}"
            st.markdown(answer)
            if learning_instructions(feedback_profile):
                with st.expander("Saved feedback applied"):
                    for instruction in learning_instructions(feedback_profile):
                        st.write(f"- {instruction}")
            match_label = (
                "Strong" if result.knowledge_match >= 0.45 else "Partial" if result.knowledge_match >= 0.20 else "Weak"
            )
            st.caption(f"{answer_engine} · Knowledge match: {match_label}")
            if result.sources:
                with st.expander("Sources used"):
                    for source in result.sources:
                        st.write(f"- `{source}`")

        chat_messages.append(
            {
                "role": "assistant",
                "content": answer,
                "engine": answer_engine,
                "match": match_label,
                "sources": result.sources,
            }
        )

elif mode == "Assist Mode":
    st.markdown('<div class="section-kicker">Decision support</div>', unsafe_allow_html=True)
    st.subheader("Ask the former expert")
    st.write(f"Describe a client issue, engineering problem, report, presentation, or company task. SkillVault will use {COMPANY_NAME}'s expert knowledge, cite its sources, and flag when escalation is needed.")

    default_case_text = (
        "A client reports that file uploads fail for files over 100 MB, while smaller files work. What should I do and what should I tell the client?"
        if IS_NORTHSTAR_DEMO
        else ""
    )
    case_text = st.text_area(
        "Task or client issue",
        value=default_case_text,
        placeholder="Describe the current situation, what changed, who is affected, and the result you need.",
        height=150,
    )
    st.session_state.setdefault("employee_team", "")
    assist_team = st.text_input(
        "Your team (optional)", value=st.session_state["employee_team"], key="assist_employee_team",
        help="Helps tailor the answer; your team is not assumed from the company's profile.",
    )
    st.session_state["employee_team"] = assist_team
    assist_context_file = st.file_uploader(
        "Current ticket or document (optional)",
        type=["txt", "md", "log", "csv", "json", "pdf", "docx", "pptx", "py", "js", "ts", "sql", "eml"],
        key="assist_current_file",
        help="Current-case context only. This file is not added to approved company knowledge.",
    )
    if assist_context_file:
        st.caption("The file itself is not saved as a company decision. If you save feedback or approve the resulting plan, its answer may summarize details from the file.")
    focus_col, analyze_col = st.columns([1.25, 1])
    with focus_col:
        scenario_filter = st.selectbox("Knowledge focus area", SCENARIO_OPTIONS, help="Prioritize historical decisions from one work area. Choose Other for cases outside the main categories.")
    with analyze_col:
        st.write("")
        analyze_clicked = st.button("Analyze case", type="primary", width="stretch", disabled=not case_text.strip() and assist_context_file is None)

    if analyze_clicked:
        current_case = prepare_current_case(
            case_text, assist_team,
            WorkFile(assist_context_file.name, assist_context_file.type or "application/octet-stream", assist_context_file.getvalue()) if assist_context_file else None,
        )
        if current_case.warning:
            st.warning(current_case.warning)
        result = engine.predict(current_case.retrieval_question(), scenario_filter)
        st.session_state["assist_result"] = result
        st.session_state["assist_query"] = current_case.model_question()
        st.session_state["assist_user_question"] = case_text.strip() or f"Help with attached work: {current_case.filename}"
        st.session_state["assist_current_case"] = current_case
        st.session_state["assist_scenario"] = scenario_filter
        st.session_state.pop("assist_refined_from", None)
        st.session_state.pop("assist_decision_status", None)
        st.session_state.pop("assist_answer_record", None)
        st.session_state.pop("assist_feedback_context", None)
        st.session_state.pop("assist_feedback_message", None)

    result = st.session_state.get("assist_result")
    if result is not None:
        case_text = st.session_state.get("assist_query", case_text)
        recorded_question = st.session_state.get("assist_user_question", case_text)
        confidence_class = "confidence-high" if result.confidence >= 0.75 else "confidence-mid"

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Recommended approach", result.label)
        with c2:
            st.markdown(f"<div class='metric-card'><strong>Confidence</strong><br><span class='{confidence_class}'>{result.confidence:.0%}</span></div>", unsafe_allow_html=True)
        with c3:
            st.metric("Review status", "Human review" if result.confidence < 0.75 else "Ready to review")
        st.caption(
            f"Indexed retrieval shortlisted {result.candidate_count:,} of {result.indexed_count:,} approved decisions before vector reranking · {result.retrieval_mode}"
        )

        st.divider()
        left, right = st.columns([1.15, 1])
        with left:
            st.markdown("### Why this recommendation?")
            st.write(result.explanation)
            st.markdown("### Retrieved expert playbook (source guidance)")
            st.markdown(result.guidance)
            if result.general_suggestions:
                st.warning("The uploaded expert files do not contain a close enough match. The suggestions below are general best practices, not instructions taken from the source files.")
                st.markdown("### Additional general suggestions")
                for suggestion in result.general_suggestions:
                    st.markdown(f"- {suggestion}")
            if result.confidence < 0.75:
                st.warning("This case is unlike the examples the model knows well. A human should make the final decision.")
            else:
                st.info("The model is assisting, not replacing, the employee's judgment.")
        with right:
            st.markdown("### Similar expert cases")
            for case in result.similar_cases:
                with st.container(border=True):
                    st.caption(f"Similarity {case.similarity:.0%}")
                    st.write(f"**Historical situation:** {case.summary}")
                    if case.goal:
                        st.write(f"**Goal:** {case.goal}")
                    st.write(f"**Decision pattern:** {case.label}")
                    if case.chosen_approach:
                        st.write(f"**Exact choice made:** {case.chosen_approach}")
                    st.write(f"**Why the expert chose it:** {case.reasoning}")
                    if case.alternatives_considered:
                        st.write(f"**Alternatives rejected:** {case.alternatives_considered}")
                    if case.constraints:
                        st.write(f"**Constraints:** {case.constraints}")
                    st.write(f"**Reusable steps:** {case.instructions or case.reasoning}")
                    if case.reusable_rule:
                        st.write(f"**Reusable rule:** {case.reusable_rule}")
                    if case.exceptions:
                        st.write(f"**Do not reuse / escalate when:** {case.exceptions}")
                    st.write(f"**Software/tools:** {case.software or 'Not specified'}")
                    st.write(f"**Methods:** {case.methods or 'Not specified'}")
                    if case.capture_evidence:
                        with st.expander("Original work and employee explanation"):
                            for document in case.capture_evidence.get("documents", []):
                                st.caption(f"{document.get('role', '')}: {document.get('filename', '')}")
                                st.text(str(document.get("text", ""))[:6000])
                                original_bytes = read_artifact(ARTIFACTS_PATH, str(document.get("sha256", "")))
                                if original_bytes is not None:
                                    st.download_button(
                                        "Download original file", original_bytes,
                                        file_name=Path(str(document.get("filename", "source"))).name,
                                        key=f"original_{result.similar_cases.index(case)}_{document.get('source_id')}",
                                    )
                            for citation in case.capture_evidence.get("decision_citations", []):
                                st.caption(f"{citation['filename']} · {citation.get('location', 'Extracted text line ' + str(citation['extracted_line']))}")
                                st.text(citation["quote"])
                            for field_name, confirmed_value in case.capture_evidence.get("employee_confirmed", {}).items():
                                st.write(f"**Employee confirmed — {field_name.replace('_', ' ')}:** {confirmed_value}")
                            visual = case.capture_evidence.get("visual_comparison", {})
                            if visual:
                                st.caption(f"Employee-reviewed visual changes: {visual.get('before_source')} → {visual.get('after_source')}. Not proof of intent or success.")
                                st.write(visual.get("observation", ""))
                    if case.outcome:
                        st.write(f"**Outcome:** {case.outcome}")
                    st.caption(
                        f"Source file: `{case.source_file}` · Owner: {case.expert_owner} · "
                        f"Department: {case.department} · Reviewed: {case.last_reviewed_date} · "
                        f"Expires: {case.expiration_date} · Status: {case.lifecycle_status}"
                    )
                    if case.media:
                        st.markdown("**Attached expert media**")
                        for media in case.media:
                            st.caption(str(media["name"]))
                            if str(media["type"]).startswith("image/"):
                                st.image(media["bytes"], width="stretch")
                            elif str(media["type"]).startswith("video/"):
                                st.video(media["bytes"])

            st.markdown("### Source files used")
            for source in result.sources:
                st.write(f"📄 `{source}`")

        followup = material_followup(st.session_state.get("assist_current_case", prepare_current_case(case_text)), result.knowledge_match)
        if followup and "assist_refined_from" not in st.session_state:
            if result.knowledge_match < 0.35:
                st.warning("The closest historical decision is a weak match. The answer below still gives an initial plan; this one detail could improve it.")
            clarification = st.text_area(
                followup,
                placeholder="Add only the detail you know.",
                key="assist_clarification",
                height=110,
            )
            if st.button("Update answer with this context", key="refine_assist_answer"):
                if not clarification.strip():
                    st.warning("Add at least one useful detail before updating the recommendation.")
                else:
                    refined_query = f"{case_text}\n\nAdditional context from employee: {clarification.strip()}"
                    st.session_state["assist_refined_from"] = result
                    st.session_state["assist_query"] = refined_query
                    st.session_state["assist_user_question"] = f"{recorded_question}\nAdditional context: {clarification.strip()}"
                    st.session_state["assist_result"] = engine.predict(refined_query, scenario_filter)
                    st.session_state.pop("assist_decision_status", None)
                    st.session_state.pop("assist_answer_record", None)
                    st.session_state.pop("assist_feedback_context", None)
                    st.rerun()

        if "assist_refined_from" in st.session_state:
            original = st.session_state["assist_refined_from"]
            with st.expander("See the closest decision before clarification"):
                st.caption("This is the low-similarity evidence SkillVault started with before the employee added context.")
                for old_case in original.similar_cases[:1]:
                    st.write(f"**Historical case:** {old_case.summary}")
                    st.write(f"**Original expert approach:** {old_case.label}")
                    st.write(old_case.instructions or old_case.reasoning)
                    st.caption(f"Source: `{old_case.source_file}`")

        st.markdown("### SkillVault answer")
        answer_record = st.session_state.get("assist_answer_record", {})
        feedback_context = str(st.session_state.get("assist_feedback_context", "")).strip()
        feedback_profile = learning_profile(load_recommendation_feedback(RECOMMENDATION_FEEDBACK_PATH), case_text, WORKSPACE_ID)
        generation_key = f"two-recommendations-v2\n{case_text}\n{feedback_context}\n{feedback_profile}"
        if answer_record.get("generation_key") != generation_key:
            effective_question = case_text
            if feedback_context:
                effective_question = (
                    f"Original employee question:\n{case_text}\n\n"
                    "Employee feedback about the previous recommendation:\n"
                    f"{feedback_context}\n\n"
                    "Rewrite the recommendation to address that feedback while staying grounded in approved company evidence."
                )
            generation_error = ""
            if local_model.available:
                try:
                    with st.spinner(f"Local {local_model.model} is adapting the approved decisions to this case..."):
                        generated_answer = local_model.grounded_answer(effective_question, result, company_profile=COMPANY_PROFILE, feedback_profile=feedback_profile)
                    answer_engine = f"Local {local_model.model}"
                except LocalModelError as error:
                    generated_answer = engine.compose_response(effective_question, result, feedback_profile=feedback_profile)
                    answer_engine = "Local evidence-grounded fallback"
                    generation_error = str(error)
            else:
                generated_answer = engine.compose_response(effective_question, result, feedback_profile=feedback_profile)
                answer_engine = "Local evidence-grounded fallback"
            answer_record = {
                "query": case_text,
                "generation_key": generation_key,
                "answer": generated_answer,
                "engine": answer_engine,
                "error": generation_error,
            }
            st.session_state["assist_answer_record"] = answer_record

        final_answer = str(answer_record.get("answer", ""))
        if answer_record.get("error"):
            st.warning(f"The configured model was unavailable, so SkillVault used its local fallback: {answer_record['error']}")
        st.caption(
            f"Written by {answer_record.get('engine', 'Local evidence-grounded fallback')}. "
            "The answer combines retrieved approved decisions, recommends useful actions before escalation, and clearly labels "
            "independent guidance. It does not send private case data to a public internet search."
        )
        st.markdown(final_answer)
        if learning_instructions(feedback_profile):
            with st.expander("How saved feedback influenced this answer"):
                st.caption("Saved answer preferences—not model-weight training or new company policy.")
                for instruction in learning_instructions(feedback_profile):
                    st.write(f"- {instruction}")

        st.markdown("### Decision status")
        approve_col, review_col = st.columns(2)
        with approve_col:
            if st.button("Approve final plan", type="primary", key="approve_final_plan"):
                approved_lineage = engine.build_decision_lineage(
                    recorded_question,
                    result,
                    approval_status="Approved",
                )
                plan_id = save_approved_plan(
                    APPROVED_PLANS_PATH,
                    recorded_question,
                    result.label,
                    result.confidence,
                    result.sources,
                    final_answer,
                    lineage=approved_lineage,
                )
                st.session_state["assist_decision_status"] = {
                    "query": case_text,
                    "status": "Approved",
                    "plan_id": plan_id,
                }
                st.success("Final plan approved and saved to the local decision history.")
        with review_col:
            if st.button("Mark for expert review", key="mark_plan_review"):
                st.session_state["assist_decision_status"] = {
                    "query": case_text,
                    "status": "Expert review requested",
                    "plan_id": "",
                }
                st.info("Marked for expert review. Keep the sources and missing context with the handoff.")

        decision_status = st.session_state.get("assist_decision_status", {})
        if decision_status.get("query") == case_text:
            lineage_status = decision_status.get("status", "Awaiting approval")
            lineage_plan_id = decision_status.get("plan_id", "")
        else:
            lineage_status = "Awaiting approval"
            lineage_plan_id = ""
        lineage = engine.build_decision_lineage(
            recorded_question,
            result,
            approval_status=lineage_status,
            plan_id=lineage_plan_id,
        )
        render_decision_lineage(lineage)

        st.markdown("### Improve this recommendation")
        st.write(
            "Tell SkillVault exactly what worked, what was wrong, and what it should do differently. "
            "You can save the feedback for review or immediately regenerate the answer with your corrections."
        )
        feedback_message = st.session_state.pop("assist_feedback_message", "")
        if feedback_message:
            st.success(feedback_message)

        can_approve_feedback = str(CURRENT_USER.get("role", "")).lower() in {
            "administrator",
            "admin",
            "expert",
            "demo",
        }
        with st.form("recommendation_feedback_form"):
            verdict = st.segmented_control(
                "Overall result",
                ["Helpful", "Needs improvement", "Incorrect or unsafe"],
                default="Needs improvement",
                key="feedback_verdict",
            )
            relevance_col, clarity_col, action_col = st.columns(3)
            with relevance_col:
                relevance_rating = st.slider("Relevance", 1, 5, 3, key="feedback_relevance")
            with clarity_col:
                clarity_rating = st.slider("Clarity", 1, 5, 3, key="feedback_clarity")
            with action_col:
                actionability_rating = st.slider("Actionability", 1, 5, 3, key="feedback_actionability")

            issue_types = st.multiselect(
                "What should be improved?",
                [
                    "Wrong past decision or source",
                    "Too generic",
                    "Too vague",
                    "Not specific enough",
                    "Too short",
                    "Too long",
                    "Not helpful",
                    "Missing important steps",
                    "Recommended the wrong tool or method",
                    "Escalated too early",
                    "Did not use the facts I provided",
                    "Outdated or unsafe guidance",
                    "Unclear explanation",
                    "Other",
                ],
                key="feedback_issue_types",
            )
            learn_for_future = st.checkbox("Use this feedback to improve future answers in this company", value=True, key="feedback_learn_future")
            st.caption("Selected issues and low clarity/actionability ratings adjust future answer style. Negative feedback also triggers extra checks for related questions. Free-text corrections still need expert approval to become company knowledge.")
            reviewer_notes = st.text_area(
                "What was good or wrong about the recommendation?",
                placeholder="Example: The source was relevant, but the answer skipped checking the gateway limit and escalated before giving the support engineer anything useful to test.",
                height=100,
                key="feedback_notes",
            )
            missing_context = st.text_area(
                "Add facts the answer missed or misunderstood",
                placeholder="Example: Only one tenant is affected, 99 MB works, 101 MB fails, and no production setting has been changed.",
                height=90,
                key="feedback_missing_context",
            )
            suggested_steps = st.text_area(
                "What steps should the improved answer recommend?",
                placeholder="Example: Reproduce both sizes, compare client/gateway/server limits, inspect the rejected request, then change only the confirmed failing layer and retest.",
                height=120,
                key="feedback_suggested_steps",
            )

            st.markdown("#### Optional expert correction")
            if can_approve_feedback:
                expert_approved = st.checkbox(
                    "I am confirming these corrected steps as approved company knowledge",
                    key="feedback_expert_approved",
                )
                correction_label = st.selectbox(
                    "Correct decision pattern",
                    engine.labels,
                    index=engine.labels.index(result.label) if result.label in engine.labels else 0,
                    key="feedback_correction_label",
                )
                expert_owner = st.text_input(
                    "Expert owner",
                    value=str(CURRENT_USER.get("display_name", "Assist Mode reviewer")),
                    key="feedback_expert_owner",
                )
                st.caption("Only checked expert corrections become searchable company knowledge. Opted-in feedback also adjusts future answer-writing preferences.")
            else:
                expert_approved = False
                correction_label = result.label
                expert_owner = ""
                st.caption("Your feedback will be saved for an administrator or expert to review. It will not automatically become company policy.")

            save_col, regenerate_col = st.columns(2)
            with save_col:
                save_feedback = st.form_submit_button(
                    "Save feedback",
                    icon=":material/save:",
                    width="stretch",
                )
            with regenerate_col:
                regenerate_answer = st.form_submit_button(
                    "Save and improve answer",
                    type="primary",
                    icon=":material/refresh:",
                    width="stretch",
                )

        if save_feedback or regenerate_answer:
            has_explanation = bool(issue_types or reviewer_notes.strip() or missing_context.strip() or suggested_steps.strip())
            validation_errors = []
            if verdict != "Helpful" and not has_explanation:
                validation_errors.append("Describe at least one problem or missing fact so the feedback is useful.")
            if expert_approved and not suggested_steps.strip():
                validation_errors.append("An expert correction needs specific corrected steps before it can become company knowledge.")
            if expert_approved and not expert_owner.strip():
                validation_errors.append("Enter the expert owner approving the correction.")

            if validation_errors:
                for validation_error in validation_errors:
                    st.error(validation_error)
            else:
                feedback_payload = {
                    "workspace_id": WORKSPACE_ID,
                    "company_name": COMPANY_NAME,
                    "query": recorded_question,
                    "recommendation_snapshot": final_answer,
                    "answer_engine": str(answer_record.get("engine", "")),
                    "retrieved_sources": list(result.sources),
                    "knowledge_match": result.knowledge_match,
                    "verdict": verdict,
                    "ratings": {
                        "relevance": relevance_rating,
                        "clarity": clarity_rating,
                        "actionability": actionability_rating,
                    },
                    "issue_types": issue_types,
                    "learn_for_future": learn_for_future,
                    "reviewer_notes": reviewer_notes.strip(),
                    "missing_context": missing_context.strip(),
                    "suggested_steps": suggested_steps.strip(),
                    "reviewer": {
                        "user_id": str(CURRENT_USER.get("user_id", "")),
                        "display_name": str(CURRENT_USER.get("display_name", "")),
                        "email": str(CURRENT_USER.get("email", "")),
                        "role": str(CURRENT_USER.get("role", "")),
                    },
                    "expert_approved": expert_approved,
                    "corrected_decision_pattern": correction_label if expert_approved else "",
                }
                feedback_id = save_recommendation_feedback(RECOMMENDATION_FEEDBACK_PATH, feedback_payload)

                if expert_approved:
                    correction_case = ExpertCase(
                        recorded_question,
                        correction_label,
                        reviewer_notes.strip() or "An authorized expert corrected the previous recommendation.",
                        infer_features(recorded_question),
                        f"{feedback_id}.md",
                        suggested_steps.strip(),
                        f"{COMPANY_NAME} approved systems",
                        "Expert review; recommendation correction",
                        "Approved correction saved for future retrieval and outcome verification.",
                        expert_owner=expert_owner.strip(),
                        department=str(COMPANY_PROFILE.get("primary_team", "Company operations")),
                        approval_date=date.today().isoformat(),
                        last_reviewed_date=date.today().isoformat(),
                        expiration_date=(date.today() + timedelta(days=365)).isoformat(),
                        lifecycle_status="Current",
                        goal=f"Correct future SkillVault guidance for: {recorded_question[:180]}",
                        chosen_approach=suggested_steps.strip().splitlines()[0][:240],
                        constraints=missing_context.strip(),
                        reusable_rule="Use the expert-corrected steps when the current facts match this reviewed situation.",
                        exceptions="Recheck the correction when current systems, policy, risk, or scope differ.",
                    )
                    append_case(APPROVED_CASES_PATH, correction_case)
                    get_engine.clear()

                if regenerate_answer:
                    feedback_parts = [
                        f"Overall feedback: {verdict}.",
                        f"Problems selected: {', '.join(issue_types)}." if issue_types else "",
                        f"Reviewer explanation: {reviewer_notes.strip()}" if reviewer_notes.strip() else "",
                        f"New or corrected facts: {missing_context.strip()}" if missing_context.strip() else "",
                        f"Preferred steps: {suggested_steps.strip()}" if suggested_steps.strip() else "",
                    ]
                    feedback_context = "\n".join(part for part in feedback_parts if part)
                    st.session_state["assist_feedback_context"] = feedback_context
                    st.session_state["assist_result"] = engine.predict(
                        f"{case_text}\n{feedback_context}",
                        scenario_filter,
                    )
                    st.session_state.pop("assist_answer_record", None)
                    st.session_state.pop("assist_decision_status", None)
                    st.session_state["assist_feedback_message"] = (
                        f"Feedback {feedback_id} was saved. SkillVault regenerated the recommendation using your corrections."
                    )
                    st.rerun()
                elif expert_approved:
                    st.session_state["assist_feedback_message"] = (
                        f"Feedback {feedback_id} was saved and the expert correction was added to approved company knowledge."
                    )
                    st.rerun()
                else:
                    st.success(f"Feedback {feedback_id} was saved. " + ("Answer preferences will apply to future responses in this company. " if learn_for_future else "Future-answer learning was disabled for this feedback. ") + "No unapproved facts were added to company knowledge.")

elif mode == "Add Company Knowledge" and knowledge_method == "Upload completed work":
    st.markdown("### Upload the work and capture the decision behind it")
    st.write("Upload the finished work itself—such as a PowerPoint, code files, report, ticket, or runbook. SkillVault reads the artifact, reconstructs a structured decision, flags reasoning the artifact cannot prove, and requires expert approval before retrieval.")

    with st.container(border=True):
        st.markdown("**How capture works**")
        st.write("1. Upload completed work.  2. SkillVault extracts what happened.  3. The employee confirms why that choice was made and when it should be reused.  4. An expert approves the decision record.")
        st.caption("The stored knowledge is not just the file. It is: situation + goal + exact choice + reasoning + rejected alternatives + constraints + reusable steps + outcome + exceptions + source evidence.")

    artifact_type = st.selectbox(
        "What kind of completed work is this?",
        ["Auto-detect", "Presentation or pitch deck", "Code or technical implementation", "Support or incident resolution", "Report or analysis", "Process or policy document", "Other"],
    )

    capture_mode = st.segmented_control(
        "What are you sharing?",
        ["Completed work", "Before and after", "Explain what happened"],
        default="Completed work",
    )
    capture_types = ["txt", "md", "log", "csv", "json", "yaml", "yml", "toml", "ini", "xml", "rst", "patch", "diff", "ipynb", "py", "js", "jsx", "ts", "tsx", "java", "cs", "go", "rb", "php", "sh", "ps1", "html", "css", "sql", "eml", "pdf", "docx", "pptx", "png", "jpg", "jpeg", "webp", "mp4", "mov", "webm"]
    before_files = []
    if capture_mode == "Before and after":
        before_files = st.file_uploader(
            "Before: original work", type=capture_types, accept_multiple_files=True, key="capture_before",
            help="The original deck, code, report, or process before the change.",
        )
    completed_files = st.file_uploader(
        "After: revised work" if capture_mode == "Before and after" else "Upload completed-work evidence",
        type=capture_types,
        accept_multiple_files=True,
        key="capture_after",
        help="Examples: completed PowerPoint deck, support ticket export, email, incident report, logs, pull-request notes, PDF, Word file, screenshot, or screen recording.",
    )
    supporting_files = st.file_uploader(
        "Supporting evidence (optional)", type=capture_types, accept_multiple_files=True,
        key="capture_supporting", help="Approval email, review comments, PR description, test result, or outcome report.",
    )
    st.caption("For the least typing, upload the final artifact together with its decision trail—for example a deck + speaker notes/approval email, or code + PR description/diff/test results. SkillVault combines all uploaded evidence into one draft.")
    confirmed_recording_files = []
    with st.expander("Explain the decision with a recording"):
        st.write("Upload a short voice memo or the audio from a screen recording. Review the transcript, then include it as supporting evidence.")
        recording = st.file_uploader("Decision recording", type=["wav", "mp3", "m4a", "ogg", "flac", "mp4", "webm"], key="decision_recording")
        recording_digest = sha256(recording.getvalue()).hexdigest() if recording else ""
        if st.button("Transcribe recording locally", disabled=recording is None):
            try:
                with st.spinner("Transcribing the recording on this computer…"):
                    transcript = transcribe_recording(recording.getvalue(), app_setting("SKILLVAULT_SPEECH_MODEL", "base"))
                st.session_state["capture_transcript"] = {"digest": recording_digest, "text": transcript}
            except TranscriptionError as error:
                st.error(str(error))
        transcript_record = st.session_state.get("capture_transcript", {})
        if recording and transcript_record.get("digest") == recording_digest:
            reviewed_transcript = st.text_area("Review and correct the transcript", value=transcript_record["text"], height=180, key=f"transcript_{recording_digest}")
            confirmed_transcript = st.checkbox("I reviewed this transcript and want to include it", key=f"confirm_transcript_{recording_digest}")
            if confirmed_transcript and reviewed_transcript.strip():
                confirmed_recording_files = [
                    WorkFile(recording.name, recording.type or "application/octet-stream", recording.getvalue()),
                    WorkFile(f"{recording.name}.reviewed-transcript.md", "text/markdown", reviewed_transcript.encode("utf-8")),
                ]
        st.caption("Requires the optional local speech setup described in the README. This extracts spoken words; it does not analyze video frames.")
    employee_notes = st.text_area(
        "Optional short explanation",
        placeholder="One or two sentences can help: We used this slide order because the investors cared more about customer proof than product detail, and moved technical slides to the appendix.",
        height=110,
        help="You do not need to rewrite the work. Add only context that is not visible in the artifact, especially for screenshots or video.",
    )
    with st.expander("Add any decision context you already know (optional)"):
        known_goal = st.text_input("Goal or problem being solved", placeholder="Convince seed investors that adoption is repeatable")
        known_choice = st.text_input("Exact choice that was made", placeholder="Lead with customer proof and move architecture to the appendix")
        known_reasoning = st.text_area("Why this choice was made", placeholder="The audience cared about traction and risk reduction more than implementation detail", height=90)
        known_alternatives = st.text_input("Alternatives considered", placeholder="A product-tour-first deck; a technical architecture opening")
        known_constraints = st.text_input("Constraints or tradeoffs", placeholder="Eight-minute pitch; no unverified revenue projections")
        known_outcome = st.text_input("Known result", placeholder="Investors requested a technical follow-up meeting")

    if st.button("Create expert-decision draft", type="primary"):
        if capture_mode == "Before and after" and (not before_files or not completed_files):
            st.warning("Upload both the original work and revised work so SkillVault can compare them.")
        elif not completed_files and not supporting_files and not confirmed_recording_files and not employee_notes.strip():
            st.warning("Upload at least one file or add a short note about the completed work.")
        else:
            work_files = [WorkFile(file.name, file.type or "application/octet-stream", file.getvalue()) for file in completed_files]
            decision_context = {
                "artifact_type": artifact_type,
                "goal": known_goal,
                "chosen_approach": known_choice,
                "reasoning": known_reasoning,
                "alternatives_considered": known_alternatives,
                "constraints": known_constraints,
                "outcome": known_outcome,
            }
            draft = capture_work(
                [WorkFile(file.name, file.type or "application/octet-stream", file.getvalue()) for file in before_files],
                work_files,
                [WorkFile(file.name, file.type or "application/octet-stream", file.getvalue()) for file in supporting_files] + confirmed_recording_files,
                employee_notes, decision_context,
            )
            st.session_state["capture_original_files"] = [
                WorkFile(file.name, file.type or "application/octet-stream", file.getvalue())
                for file in [*before_files, *completed_files, *supporting_files]
            ] + confirmed_recording_files
            st.session_state["capture_pending_drafts"] = []
            st.session_state["capture_batch_split"] = False
            extraction_error = ""
            has_model_readable_image = any(
                str(item.get("type", "")).startswith("image/")
                for item in draft.media
            )
            if local_model.available and (draft.evidence_preview.strip() or has_model_readable_image):
                try:
                    with st.spinner(f"Local {local_model.model} is reconstructing the expert decision from the artifact..."):
                        draft = local_model.enrich_decision_draft(draft, artifact_type)
                except LocalModelError as error:
                    extraction_error = str(error)
            st.session_state["completed_work_draft"] = draft
            st.session_state["capture_revision"] = st.session_state.get("capture_revision", 0) + 1
            st.session_state["completed_work_extraction_error"] = extraction_error

    draft = st.session_state.get("completed_work_draft")
    capture_message = st.session_state.pop("capture_saved_message", "")
    if capture_message:
        st.success(capture_message)
    if draft is not None:
        st.divider()
        st.markdown("### Review the extracted draft")
        pending_drafts = st.session_state.get("capture_pending_drafts", [])
        st.caption(f"Decisions awaiting review: {1 + len(pending_drafts)}")
        if not st.session_state.get("capture_batch_split", False):
            if st.button("Find separate decisions in this work", disabled=not local_model.available):
                try:
                    with st.spinner("Finding distinct decisions and verifying their source quotations..."):
                        proposals = local_model.split_decision_drafts(draft, artifact_type)
                    st.session_state["completed_work_draft"] = proposals[0]
                    st.session_state["capture_pending_drafts"] = proposals[1:]
                    st.session_state["capture_batch_split"] = True
                    st.session_state["capture_revision"] = st.session_state.get("capture_revision", 0) + 1
                    st.rerun()
                except LocalModelError as error:
                    st.error(str(error))
            if not local_model.available:
                st.caption("Connect the local language model to find multiple decisions automatically. You can still review this draft.")
        if st.button("Discard this draft", key="discard_capture_draft"):
            if pending_drafts:
                st.session_state["completed_work_draft"] = pending_drafts[0]
                st.session_state["capture_pending_drafts"] = pending_drafts[1:]
            else:
                st.session_state.pop("completed_work_draft", None)
                st.session_state.pop("capture_original_files", None)
            st.session_state["capture_revision"] = st.session_state.get("capture_revision", 0) + 1
            st.rerun()
        if draft.capture_evidence.get("decision_citations"):
            with st.expander("Evidence for this decision", expanded=True):
                for citation in draft.capture_evidence["decision_citations"]:
                    st.caption(f"{citation['filename']} · {citation.get('location', 'Extracted text line ' + str(citation['extracted_line']))}")
                    st.text(citation["quote"])
        st.caption("This draft is an extraction aid, not approved knowledge. Correct anything that is incomplete or inferred.")
        extraction_error = st.session_state.get("completed_work_extraction_error", "")
        if extraction_error:
            st.warning(f"The configured model could not structure this artifact, so the local extractor created the draft: {extraction_error}")
        st.caption(f"Drafted by: {draft.generation_method}")
        for warning in draft.warnings:
            st.warning(warning)
        if draft.missing_context:
            st.info("The artifact did not establish: " + "; ".join(draft.missing_context) + ". Confirm these fields below before approval.")

        if draft.capture_evidence.get("comparison"):
            with st.expander("What changed between before and after", expanded=True):
                st.code(str(draft.capture_evidence["comparison"]), language="diff")
        if draft.capture_evidence.get("mode") == "before_after":
            originals = {sha256(file.data).hexdigest(): file for file in st.session_state.get("capture_original_files", [])}
            image_sources = {
                role: [doc for doc in draft.capture_evidence.get("documents", [])
                       if doc.get("role") == role and doc.get("sha256") in originals
                       and originals[doc["sha256"]].content_type.startswith("image/")]
                for role in ("before", "after")
            }
            if all(image_sources.values()):
                with st.expander("Compare visual changes"):
                    st.caption("Select matching screenshots. Requires a vision-capable local model. PowerPoint/PDF files are not rendered automatically; upload screenshots of the relevant slides or pages.")
                    revision = st.session_state.get("capture_revision", 0)
                    selected_before = st.selectbox("Original screenshot", image_sources["before"], format_func=lambda doc: doc["filename"], key=f"visual_before_{revision}")
                    selected_after = st.selectbox("Revised screenshot", image_sources["after"], format_func=lambda doc: doc["filename"], key=f"visual_after_{revision}")
                    pair_key = f"visual_{revision}_{selected_before['sha256']}_{selected_after['sha256']}"
                    st.image(originals[selected_before["sha256"]].data, caption="Before")
                    st.image(originals[selected_after["sha256"]].data, caption="After")
                    if st.button("Describe visible changes", disabled=not local_model.available):
                        try:
                            with st.spinner("Comparing screenshots locally..."):
                                st.session_state[pair_key] = local_model.compare_work_images(originals[selected_before["sha256"]], originals[selected_after["sha256"]])
                        except LocalModelError as error:
                            st.error(str(error))
                    if pair_key in st.session_state:
                        edited_observation = st.text_area("Review the visible-change description", value=st.session_state[pair_key], key=f"edit_{pair_key}")
                        confirmed_visual = st.checkbox("I checked this description against both images", key=f"confirm_{pair_key}")
                        if st.button("Attach reviewed visual evidence", disabled=not confirmed_visual or not edited_observation.strip()):
                            draft.capture_evidence["visual_comparison"] = {
                                "before_source": selected_before["source_id"], "after_source": selected_after["source_id"],
                                "observation": edited_observation.strip(), "reviewed_by_employee": True,
                            }
                            marker = "\n\nEMPLOYEE-REVIEWED VISUAL OBSERVATION (not proof of intent or outcome):\n"
                            draft.evidence_preview = draft.evidence_preview.split(marker)[0] + marker + edited_observation.strip()
                            st.session_state["completed_work_draft"] = draft
                            st.success("Attached to this draft. It becomes company knowledge only when you approve the decision below.")
                    if draft.capture_evidence.get("visual_comparison"):
                        st.write(draft.capture_evidence["visual_comparison"]["observation"])
        next_question = next_interview_question(draft)
        if next_question:
            st.markdown("### One detail the files could not establish")
            st.caption("Answer only if you know. Another missing detail may appear after this; you can also finish in the review form. Do not guess an outcome.")
            revision = st.session_state.get("capture_revision", 0)
            with st.form(f"capture_interview_{revision}"):
                field, question = next_question
                interview_answer = st.text_area(question, key=f"interview_{revision}_{field}")
                interview_submitted = st.form_submit_button("Add answers to my decision", type="primary")
            if interview_submitted:
                if interview_answer.strip():
                    st.session_state["completed_work_draft"] = apply_interview(draft, {field: interview_answer})
                    st.session_state["capture_revision"] = revision + 1
                    st.rerun()
                else:
                    st.info("Add an answer, or continue to the editable review below.")

        metric_one, metric_two, metric_three = st.columns(3)
        metric_one.metric("Readable sources", len(draft.readable_files))
        metric_two.metric("Media attachments", len(draft.media))
        metric_three.metric("Draft approach", draft.decision)

        if draft.evidence_preview:
            with st.expander("Inspect extracted source text"):
                st.text(draft.evidence_preview)
        if draft.media:
            with st.expander("Inspect attached screenshots and recordings"):
                for media_item in draft.media:
                    st.caption(str(media_item["name"]))
                    if str(media_item["type"]).startswith("image/"):
                        st.image(media_item["bytes"], width="stretch")
                    elif str(media_item["type"]).startswith("video/"):
                        st.video(media_item["bytes"])

        with st.form(f"completed_work_review_form_{st.session_state.get('capture_revision', 0)}"):
            imported_summary = st.text_area("Detected situation", value=draft.summary, height=100)
            decision_options = sorted(VALID_DECISIONS)
            imported_decision = st.selectbox("Detected expert approach", decision_options, index=decision_options.index(draft.decision))
            imported_goal = st.text_area("Goal or problem the expert was solving", value=draft.goal, height=90)
            imported_chosen_approach = st.text_area("Exact choice that was made", value=draft.chosen_approach, height=90)
            imported_reasoning = st.text_area("Why the expert chose this approach", value=draft.reasoning, height=125)
            imported_alternatives = st.text_area("Alternatives considered or rejected", value=draft.alternatives_considered, height=100, help="Leave blank only if there genuinely were no meaningful alternatives.")
            imported_constraints = st.text_area("Constraints and tradeoffs", value=draft.constraints, height=100)
            imported_instructions = st.text_area("Exact reusable steps", value=draft.instructions, height=190)
            imported_software = st.text_input("Detected software and tools", value=draft.software)
            imported_methods = st.text_input("Detected methods", value=draft.methods)
            imported_outcome = st.text_area("Detected outcome", value=draft.outcome, height=100)
            imported_reusable_rule = st.text_area("Reusable rule for future employees", value=draft.reusable_rule, height=90, placeholder="Use this approach when...")
            imported_exceptions = st.text_area("Exceptions, stop conditions, or when to escalate", value=draft.exceptions, height=90)
            imported_source = st.text_input("Source files", value=draft.source_file)
            owner_col, department_col = st.columns(2)
            with owner_col:
                imported_owner = st.text_input("Expert owner", value=f"{COMPANY_NAME} Knowledge Council")
            with department_col:
                imported_department = st.text_input("Department", value="Support and engineering")
            approval_col, review_col, expiration_col = st.columns(3)
            with approval_col:
                imported_approval_date = st.date_input("Approval date", value=date.today())
            with review_col:
                imported_review_date = st.date_input("Last reviewed date", value=date.today())
            with expiration_col:
                imported_expiration_date = st.date_input("Expiration date", value=date.today() + timedelta(days=365))
            imported_status = st.selectbox("Knowledge status", LIFECYCLE_STATUSES)
            imported_replaces = st.text_input("Replaces source (optional)", placeholder="Older runbook or decision this makes obsolete")
            import_approved = st.checkbox("I reviewed the source evidence and confirm this draft is accurate.")
            import_submitted = st.form_submit_button("Approve decision and update company knowledge", type="primary")

        if import_submitted:
            required_values = {
                "situation": imported_summary,
                "goal": imported_goal,
                "exact choice": imported_chosen_approach,
                "reasoning": imported_reasoning,
                "steps": imported_instructions,
                "outcome": imported_outcome,
                "source files": imported_source,
                "expert owner": imported_owner,
                "department": imported_department,
            }
            missing = [name for name, value in required_values.items() if not value.strip()]
            unresolved_markers = ("must add", "must record", "replace these draft steps")
            unresolved = any(marker in value.lower() for value in (imported_reasoning, imported_instructions, imported_outcome) for marker in unresolved_markers)
            if missing:
                st.error(f"Please complete: {', '.join(missing)}")
            elif unresolved:
                st.error("Replace the marked placeholder text with the expert's actual reasoning, steps, and outcome before approval.")
            elif not import_approved:
                st.warning("An expert must confirm the draft before SkillVault can learn it.")
            else:
                imported_case = ExpertCase(
                    imported_summary,
                    imported_decision,
                    imported_reasoning,
                    infer_features(f"{imported_summary} {imported_reasoning} {imported_instructions}"),
                    imported_source,
                    imported_instructions,
                    imported_software,
                    imported_methods,
                    imported_outcome,
                    draft.media,
                    imported_owner,
                    imported_department,
                    imported_approval_date.isoformat(),
                    imported_review_date.isoformat(),
                    imported_expiration_date.isoformat(),
                    imported_status,
                    imported_replaces,
                    goal=imported_goal,
                    chosen_approach=imported_chosen_approach,
                    alternatives_considered=imported_alternatives,
                    constraints=imported_constraints,
                    reusable_rule=imported_reusable_rule,
                    exceptions=imported_exceptions,
                    capture_evidence=draft.capture_evidence,
                )
                store_artifacts(ARTIFACTS_PATH, st.session_state.get("capture_original_files", []))
                append_case(APPROVED_CASES_PATH, imported_case)
                get_engine.clear()
                if pending_drafts:
                    st.session_state["completed_work_draft"] = pending_drafts[0]
                    st.session_state["capture_pending_drafts"] = pending_drafts[1:]
                else:
                    st.session_state.pop("completed_work_draft", None)
                    st.session_state.pop("capture_original_files", None)
                st.session_state["capture_revision"] = st.session_state.get("capture_revision", 0) + 1
                st.session_state["capture_saved_message"] = "Decision approved and its original files preserved. " + (
                    "Review the next decision below." if pending_drafts else "You can now ask SkillVault about this work."
                )
                st.rerun()

elif mode == "Add Expert Decision":
    st.markdown('<div class="section-kicker">Knowledge capture</div>', unsafe_allow_html=True)
    st.subheader("Record a completed expert action")
    st.write("Add a decision after the work happened—even days later. SkillVault will only learn it after you confirm that the entry is accurate.")

    with st.form("expert_decision_form"):
        summary = st.text_area("What situation or task happened?", placeholder="A client could not log in after enabling SSO...")
        goal = st.text_area("What goal or problem was the expert solving?", placeholder="Restore access without weakening organization-wide SSO security...")
        decision = st.selectbox("What approach did the expert use?", sorted(VALID_DECISIONS))
        chosen_approach = st.text_area("What exact choice was made?", placeholder="Keep SSO enabled, collect a SAML trace, and route the tenant configuration to the identity specialist...")
        reasoning = st.text_area("Why was that approach chosen?", placeholder="The issue affected the whole organization, so it needed identity-specialist review...")
        alternatives_considered = st.text_area("What alternatives were considered or rejected?", placeholder="Disabling SSO was rejected because it would weaken the tenant's approved security configuration...")
        constraints = st.text_area("What constraints or tradeoffs shaped the choice?", placeholder="Organization-wide impact; least-privilege policy; client update deadline...")
        instructions = st.text_area("What exact steps should someone follow next time?", placeholder="1. Capture the tenant ID...\n2. Check the identity-provider logs...\n3. Escalate with...")
        software = st.text_input("Software and tools used", placeholder=f"{COMPANY_NAME} systems; identity provider; ticketing tool")
        methods = st.text_input("Methods used", placeholder="SAML trace review; tenant isolation; least-privilege checks")
        source_file = st.text_input("Source file or document name", value="new_expert_decision.md")
        outcome = st.text_area("What was the outcome?", placeholder="Identity engineering fixed the tenant configuration...")
        reusable_rule = st.text_area("What principle should future employees reuse?", placeholder="For organization-wide identity failures, preserve the approved security configuration and escalate with complete trace evidence.")
        exceptions = st.text_area("When should this not be reused, or when must someone escalate?", placeholder="Do not use for a single user's forgotten password; escalate immediately if all tenants are affected.")
        owner_col, department_col = st.columns(2)
        with owner_col:
            expert_owner = st.text_input("Expert owner", placeholder="Avery Chen")
        with department_col:
            department = st.text_input("Department", placeholder="Technical Support")
        approval_col, review_col, expiration_col = st.columns(3)
        with approval_col:
            approval_date = st.date_input("Approval date", value=date.today())
        with review_col:
            last_reviewed_date = st.date_input("Last reviewed date", value=date.today())
        with expiration_col:
            expiration_date = st.date_input("Expiration date", value=date.today() + timedelta(days=365))
        lifecycle_status = st.selectbox("Knowledge status", LIFECYCLE_STATUSES)
        replaces_source = st.text_input("Replaces source (optional)", placeholder="Name of an older decision this supersedes")
        media_files = st.file_uploader(
            "Attach screenshots, diagrams, or screen recordings",
            type=["png", "jpg", "jpeg", "webp", "mp4", "mov", "webm"],
            accept_multiple_files=True,
            help="Related images and videos will appear in Assist Mode when this expert case is retrieved.",
        )
        approved = st.checkbox("I confirm this is an accurate, approved expert decision.")
        submitted = st.form_submit_button("Save approved company decision", type="primary")

    if submitted:
        missing = [name for name, value in [("situation", summary), ("goal", goal), ("exact choice", chosen_approach), ("reasoning", reasoning), ("instructions", instructions), ("source file", source_file), ("outcome", outcome), ("expert owner", expert_owner), ("department", department)] if not value.strip()]
        if missing:
            st.error(f"Please complete: {', '.join(missing)}")
        elif not approved:
            st.warning("Please confirm that the expert decision is accurate before adding it.")
        else:
            media = [{"name": file.name, "type": file.type or "application/octet-stream", "bytes": file.getvalue()} for file in media_files]
            new_case = ExpertCase(
                summary,
                decision,
                reasoning,
                infer_features(summary),
                source_file,
                instructions,
                software,
                methods,
                outcome,
                media,
                expert_owner,
                department,
                approval_date.isoformat(),
                last_reviewed_date.isoformat(),
                expiration_date.isoformat(),
                lifecycle_status,
                replaces_source,
                goal=goal,
                chosen_approach=chosen_approach,
                alternatives_considered=alternatives_considered,
                constraints=constraints,
                reusable_rule=reusable_rule,
                exceptions=exceptions,
            )
            append_case(APPROVED_CASES_PATH, new_case)
            if new_case.replaces_source:
                engine.replace_cases(current_company_cases())
            elif decision_is_retrievable(new_case):
                engine.add_expert_case(new_case)
            st.success("Expert decision saved with ownership and lifecycle metadata. Current knowledge is available immediately in Assist Mode.")

elif mode == "Add Company Knowledge" and knowledge_method == "Import prepared CSV":
    st.markdown("### Import a prepared knowledge spreadsheet")
    st.write("Upload support tickets, runbooks, reports, code guidance, client templates, or case histories. Review the extracted cases, then train SkillVault only after approval.")

    template_rows = [
        {"case": "Client upload fails above 100 MB", "decision": "Diagnose and verify", "goal": "Identify the failing layer without changing unrelated upload behavior", "chosen_approach": "Boundary-test uploads and compare limits across the client, gateway, and server", "reasoning": "Reproduce the boundary and inspect client, API, server, and log evidence", "alternatives_considered": "Raising every upload limit without isolating the failing layer was rejected", "constraints": "Existing tenant limits; production-change approval; large-file timeout risk", "instructions": "Reproduce 99 MB and 101 MB uploads, inspect limits and logs, then document the smallest failing request", "reusable_rule": "For size-dependent failures, isolate the smallest failing boundary before changing limits", "exceptions": "Escalate if multiple tenants are affected or a production limit must change", "source_file": "upload_incident_2025.md", "software": "Northstar Cloud Console; Northstar Cloud API Gateway; Datadog", "methods": "Boundary testing; log correlation; request tracing", "outcome": "Issue isolated", "expert_owner": "Avery Chen", "department": "Technical Support", "approval_date": date.today().isoformat(), "last_reviewed_date": date.today().isoformat(), "expiration_date": (date.today() + timedelta(days=365)).isoformat(), "lifecycle_status": "Current", "replaces_source": ""},
        {"case": "Weekly client health report needs formatting", "decision": "Follow standard process", "goal": "Give account leaders a decision-ready weekly view", "chosen_approach": "Use the approved account summary, risks, metrics, and owned next actions", "reasoning": "Use the standard account summary, risks, metrics, and next actions structure", "alternatives_considered": "A raw dashboard export was rejected because it did not explain risks or ownership", "constraints": "Executive reading time; verified metrics only; one accountable owner per action", "instructions": "Start with status, separate wins from risks, include dated metrics, and assign next-action owners", "reusable_rule": "Lead recurring reports with the decision and end with owned actions", "exceptions": "Escalate contradictory source metrics to the data owner before publishing", "source_file": "reporting_standards.md", "software": "Northstar Cloud Analytics; Snowflake; Looker", "methods": "Metric reconciliation; timezone checks; cohort analysis", "outcome": "Report approved", "expert_owner": "Jordan Lee", "department": "Customer Success", "approval_date": date.today().isoformat(), "last_reviewed_date": date.today().isoformat(), "expiration_date": (date.today() + timedelta(days=365)).isoformat(), "lifecycle_status": "Current", "replaces_source": ""},
    ]
    if not IS_NORTHSTAR_DEMO:
        template_rows = [{
            "case": "",
            "decision": "Follow standard process",
            "goal": "",
            "chosen_approach": "",
            "reasoning": "",
            "alternatives_considered": "",
            "constraints": "",
            "instructions": "",
            "reusable_rule": "",
            "exceptions": "",
            "source_file": "",
            "software": "",
            "methods": "",
            "outcome": "",
            "expert_owner": "",
            "department": "",
            "approval_date": date.today().isoformat(),
            "last_reviewed_date": date.today().isoformat(),
            "expiration_date": (date.today() + timedelta(days=365)).isoformat(),
            "lifecycle_status": "Current",
            "replaces_source": "",
        }]
    template = pd.DataFrame(template_rows)
    st.download_button("Download CSV template", template.to_csv(index=False), "skillvault_expert_cases_template.csv", "text/csv")
    uploaded = st.file_uploader("Upload expert cases (.csv)", type=["csv"])

    if uploaded is not None:
        try:
            uploaded_frame = pd.read_csv(uploaded)
            st.markdown("### Review uploaded data")
            st.caption("Required columns: case, decision, reasoning. Recommended decision-context columns: goal, chosen_approach, alternatives_considered, constraints, reusable_rule, and exceptions. Ownership, source, outcome, tools, and lifecycle columns are also supported.")
            edited_frame = st.data_editor(uploaded_frame, width="stretch", num_rows="dynamic")
            imported_cases, errors = cases_from_dataframe(edited_frame)
            if errors:
                st.error("Please fix these issues before training:")
                for error in errors[:10]:
                    st.write(f"- {error}")
            elif st.button("Approve data and update company knowledge", type="primary"):
                active_imports = [case for case in imported_cases if decision_is_retrievable(case)]
                existing_cases = load_cases(APPROVED_CASES_PATH)
                replaced_sources = {case.replaces_source.strip().lower() for case in imported_cases if case.replaces_source.strip()}
                for existing_case in existing_cases:
                    if existing_case.source_file.strip().lower() in replaced_sources:
                        existing_case.lifecycle_status = "Replaced"
                save_cases(APPROVED_CASES_PATH, existing_cases + imported_cases)
                if replaced_sources:
                    engine.replace_cases(current_company_cases())
                elif active_imports:
                    engine.add_expert_cases(active_imports)
                st.success(f"Saved {len(imported_cases)} governed decisions; {len(active_imports)} current decisions were added to retrieval.")
                st.info("Go to Assist Mode to use the updated company knowledge.")
        except Exception as error:
            st.error(f"Could not read this CSV: {error}")

elif mode == "Knowledge Governance":
    st.markdown('<div class="section-kicker">Knowledge governance</div>', unsafe_allow_html=True)
    st.subheader("Keep expert decisions owned and current")
    st.write("SkillVault only retrieves decisions whose status is Current and whose expiration date has not passed. Outdated and replaced knowledge remains visible for audit history but cannot drive a recommendation.")

    uploaded_governed_cases = load_cases(APPROVED_CASES_PATH)
    replacement_targets = {
        case.replaces_source.strip().lower()
        for case in uploaded_governed_cases
        if case.replaces_source.strip() and decision_is_retrievable(case)
    }
    governed_cases = (build_demo_cases() if IS_NORTHSTAR_DEMO else []) + uploaded_governed_cases

    def displayed_governance_status(case: ExpertCase) -> str:
        if case.source_file.strip().lower() in replacement_targets:
            return "Replaced"
        return effective_lifecycle_status(case)

    governed_statuses = [displayed_governance_status(case) for case in governed_cases]
    governance_metrics = st.columns(4)
    governance_metrics[0].metric("Total decisions", f"{len(governed_cases):,}")
    governance_metrics[1].metric("Current", f"{governed_statuses.count('Current'):,}")
    governance_metrics[2].metric("Outdated", f"{governed_statuses.count('Outdated'):,}")
    governance_metrics[3].metric("Replaced", f"{governed_statuses.count('Replaced'):,}")

    governance_rows = pd.DataFrame(
        [
            {
                "Decision": case.summary,
                "Goal": case.goal,
                "Choice made": case.chosen_approach,
                "Reusable rule": case.reusable_rule,
                "Owner": case.expert_owner,
                "Department": case.department,
                "Source": case.source_file,
                "Approved": case.approval_date,
                "Last reviewed": case.last_reviewed_date,
                "Expires": case.expiration_date,
                "Status": displayed_governance_status(case),
                "Replaces": case.replaces_source,
            }
            for case in governed_cases
        ]
    )
    st.dataframe(governance_rows, hide_index=True, width="stretch", height=360)

    st.markdown("### Review uploaded company decisions")
    governance_message = st.session_state.pop("governance_message", "")
    if governance_message:
        st.success(governance_message)
    stored_cases = load_cases(APPROVED_CASES_PATH)
    if not stored_cases:
        if IS_NORTHSTAR_DEMO:
            st.info("The built-in Northstar demonstration decisions are current. Add or import a company decision to test lifecycle editing.")
        else:
            st.info("This company workspace has no governed decisions yet. Add completed work, import a CSV, or record the first expert decision.")
    else:
        selected_case_index = st.selectbox(
            "Decision to review",
            range(len(stored_cases)),
            format_func=lambda index: f"{stored_cases[index].source_file} — {stored_cases[index].summary[:80]}",
            key="governance_case_index",
        )
        governed_case = stored_cases[selected_case_index]
        current_status = effective_lifecycle_status(governed_case)
        if current_status == "Outdated" and governed_case.lifecycle_status == "Current":
            st.warning("This decision is automatically outdated because its expiration date has passed.")

        with st.form("knowledge_governance_form"):
            owner_col, department_col = st.columns(2)
            with owner_col:
                governed_owner = st.text_input("Expert owner", value=governed_case.expert_owner)
            with department_col:
                governed_department = st.text_input("Department", value=governed_case.department)
            approval_col, review_col, expiration_col = st.columns(3)
            with approval_col:
                governed_approval = st.date_input(
                    "Approval date",
                    value=parse_iso_date(governed_case.approval_date, date.today()),
                    key="governed_approval_date",
                )
            with review_col:
                governed_review = st.date_input(
                    "Last reviewed date",
                    value=parse_iso_date(governed_case.last_reviewed_date, date.today()),
                    key="governed_review_date",
                )
            with expiration_col:
                governed_expiration = st.date_input(
                    "Expiration date",
                    value=parse_iso_date(governed_case.expiration_date, date.today() + timedelta(days=365)),
                    key="governed_expiration_date",
                )
            stored_status = governed_case.lifecycle_status if governed_case.lifecycle_status in LIFECYCLE_STATUSES else "Current"
            governed_lifecycle = st.selectbox(
                "Knowledge status",
                LIFECYCLE_STATUSES,
                index=LIFECYCLE_STATUSES.index(stored_status),
            )
            governed_replaces = st.text_input("Replaces source (optional)", value=governed_case.replaces_source)
            governance_submitted = st.form_submit_button("Save governance review", type="primary")

        if governance_submitted:
            if not governed_owner.strip() or not governed_department.strip():
                st.error("Every decision needs an expert owner and department.")
            elif governed_review < governed_approval:
                st.error("The last reviewed date cannot be earlier than the approval date.")
            elif governed_lifecycle == "Current" and governed_expiration < governed_review:
                st.error("A current decision must expire on or after its last review date.")
            else:
                governed_case.expert_owner = governed_owner.strip()
                governed_case.department = governed_department.strip()
                governed_case.approval_date = governed_approval.isoformat()
                governed_case.last_reviewed_date = governed_review.isoformat()
                governed_case.expiration_date = governed_expiration.isoformat()
                governed_case.lifecycle_status = governed_lifecycle
                governed_case.replaces_source = governed_replaces.strip()
                save_cases(APPROVED_CASES_PATH, stored_cases)
                get_engine.clear()
                st.session_state.pop("assist_result", None)
                st.session_state["governance_message"] = (
                    f"Saved {governed_case.source_file} as {effective_lifecycle_status(governed_case)}. "
                    "The retrieval index will use the updated lifecycle immediately."
                )
                st.rerun()

elif mode == "Outcome Review":
    st.markdown('<div class="section-kicker">Closed-loop learning</div>', unsafe_allow_html=True)
    st.subheader("Record what happened after the recommendation")
    st.write("Follow up on an approved SkillVault plan, record whether it worked, and optionally turn the verified result into a new governed company decision.")

    outcome_message = st.session_state.pop("outcome_message", "")
    if outcome_message:
        st.success(outcome_message)
    approved_plans = load_approved_plans(APPROVED_PLANS_PATH)
    plans_with_outcomes = [plan for plan in approved_plans if isinstance(plan.get("outcome_feedback"), dict)]
    outcome_metrics = st.columns(3)
    outcome_metrics[0].metric("Approved plans", len(approved_plans))
    outcome_metrics[1].metric("Awaiting outcome", len(approved_plans) - len(plans_with_outcomes))
    outcome_metrics[2].metric("Outcomes recorded", len(plans_with_outcomes))

    if not approved_plans:
        st.info("Approve a recommendation in Assist Mode first. It will appear here for outcome follow-up.")
    else:
        default_view = "Awaiting outcome" if len(plans_with_outcomes) < len(approved_plans) else "All plans"
        outcome_view = st.segmented_control(
            "Plans to show",
            ["Awaiting outcome", "All plans"],
            default=default_view,
            key="outcome_plan_view",
        )
        visible_plans = (
            [plan for plan in approved_plans if not isinstance(plan.get("outcome_feedback"), dict)]
            if outcome_view == "Awaiting outcome"
            else approved_plans
        )
        if not visible_plans:
            st.success("Every approved plan has outcome feedback. Switch to All plans to review or update one.")
        else:
            selected_plan_id = st.selectbox(
                "Approved plan",
                [str(plan.get("plan_id", "")) for plan in visible_plans],
                format_func=lambda plan_id: next(
                    (
                        f"{plan_id} — {str(plan.get('query', ''))[:95]}"
                        for plan in visible_plans
                        if str(plan.get("plan_id", "")) == plan_id
                    ),
                    plan_id,
                ),
                key="outcome_plan_id",
            )
            selected_plan = next(plan for plan in visible_plans if str(plan.get("plan_id", "")) == selected_plan_id)
            lineage = selected_plan.get("decision_lineage") if isinstance(selected_plan.get("decision_lineage"), dict) else {}
            existing_feedback = selected_plan.get("outcome_feedback") if isinstance(selected_plan.get("outcome_feedback"), dict) else {}

            with st.container(border=True):
                st.write(f"**Original problem:** {selected_plan.get('query', '')}")
                st.write(f"**Approved approach:** {selected_plan.get('label', '')}")
                st.caption(
                    f"Approved: {selected_plan.get('approved_at', 'Not recorded')} · "
                    f"Sources: {', '.join(selected_plan.get('sources', [])) or 'No source recorded'}"
                )
                if lineage:
                    final_action = lineage.get("final_action", {})
                    with st.expander("View the approved adapted plan"):
                        st.markdown(str(final_action.get("adapted_plan", selected_plan.get("answer", ""))))

            effectiveness_options = ["Worked", "Partially worked", "Did not work"]
            existing_effectiveness = str(existing_feedback.get("effectiveness", "Worked"))
            effectiveness_index = effectiveness_options.index(existing_effectiveness) if existing_effectiveness in effectiveness_options else 0
            already_promoted = bool(existing_feedback.get("promoted_to_knowledge"))
            with st.form("outcome_feedback_form"):
                effectiveness = st.selectbox("Did the recommendation work?", effectiveness_options, index=effectiveness_index)
                steps_changed = st.text_area(
                    "What steps were changed while carrying it out?",
                    value=str(existing_feedback.get("steps_changed", "")),
                    placeholder="Describe skipped, reordered, added, or modified steps and why.",
                    height=110,
                )
                actual_outcome = st.text_area(
                    "What was the actual outcome?",
                    value=str(existing_feedback.get("actual_outcome", "")),
                    placeholder="State the verified result, evidence, remaining risk, and any follow-up needed.",
                    height=130,
                )
                promote_to_knowledge = st.checkbox(
                    "Turn this verified result into a new approved company decision",
                    value=already_promoted,
                    disabled=already_promoted,
                )
                promotion_owner = st.text_input("Outcome reviewer / new decision owner", value=f"{COMPANY_NAME} Knowledge Council")
                promotion_department = st.text_input("Department", value="Support and engineering")
                outcome_submitted = st.form_submit_button("Save outcome feedback", type="primary")

            if outcome_submitted:
                if not actual_outcome.strip():
                    st.error("Record the actual outcome before saving the follow-up.")
                elif promote_to_knowledge and effectiveness == "Did not work":
                    st.error("A failed recommendation can be recorded, but it should not become an approved reusable decision.")
                elif promote_to_knowledge and (not promotion_owner.strip() or not promotion_department.strip()):
                    st.error("A promoted decision needs an owner and department.")
                else:
                    promoted_source = str(existing_feedback.get("promoted_source", ""))
                    if promote_to_knowledge and not already_promoted:
                        promoted_source = f"{selected_plan_id}_verified_outcome.md"
                        final_action = lineage.get("final_action", {}) if isinstance(lineage, dict) else {}
                        adapted_plan = str(final_action.get("adapted_plan", selected_plan.get("answer", ""))).strip()
                        if steps_changed.strip():
                            adapted_plan = f"{adapted_plan}\n\nExecution changes:\n{steps_changed.strip()}"
                        promoted_case = ExpertCase(
                            str(selected_plan.get("query", "")),
                            str(selected_plan.get("label", "Follow standard process")),
                            f"This approach was reviewed after execution and rated: {effectiveness}. {actual_outcome.strip()}",
                            infer_features(str(selected_plan.get("query", ""))),
                            promoted_source,
                            adapted_plan,
                            "",
                            "Approved plan execution review; outcome verification",
                            actual_outcome.strip(),
                            expert_owner=promotion_owner.strip(),
                            department=promotion_department.strip(),
                            approval_date=date.today().isoformat(),
                            last_reviewed_date=date.today().isoformat(),
                            expiration_date=(date.today() + timedelta(days=365)).isoformat(),
                            lifecycle_status="Current",
                        )
                        append_case(APPROVED_CASES_PATH, promoted_case)

                    save_plan_outcome(
                        APPROVED_PLANS_PATH,
                        selected_plan_id,
                        effectiveness,
                        steps_changed.strip(),
                        actual_outcome.strip(),
                        promote_to_knowledge,
                        promoted_source,
                    )
                    if promote_to_knowledge and not already_promoted:
                        get_engine.clear()
                    st.session_state["outcome_message"] = (
                        "Outcome saved and linked to the approved decision lineage."
                        + (" The verified result is now governed company knowledge." if promote_to_knowledge and not already_promoted else "")
                    )
                    st.rerun()

else:
    st.markdown('<div class="section-kicker">Transparency layer</div>', unsafe_allow_html=True)
    st.subheader("Model Insights")
    st.markdown("### Search by meaning")
    st.write("Semantic search helps match questions to decisions that use different wording. The first build can take a few minutes; later builds only process new or changed decisions.")
    st.caption(engine.semantic_status)
    with st.expander("Set up the local embedding model"):
        st.code("ollama pull embeddinggemma", language="powershell")
        st.caption("The embedding model is separate from the model that writes answers. You can configure SKILLVAULT_EMBEDDING_MODEL for another Ollama embedding model.")
    if st.button("Build or update semantic search", disabled=not engine.cases):
        build_progress = st.progress(0.0, text="Preparing company search index…")
        try:
            count = engine.semantic_index.build(
                engine.cases,
                progress=lambda done, total: build_progress.progress(done / max(total, 1), text=f"Embedded {done} of {total} new or changed records"),
            )
            build_progress.progress(1.0, text="Semantic search ready")
            get_engine.clear()
            st.success(f"Search index updated: {count} new or changed records. Your next question will use the updated index.")
        except SemanticSearchError as error:
            st.error(str(error))
            st.info("Keyword search still works. Start Ollama, install the embedding model, then retry; completed batches are saved.")
    st.write("A transparent view of what SkillVault has learned from the expert's historical cases.")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Expert examples", len(engine.cases))
    with c2:
        st.metric("Decision classes", len(engine.labels))
    with c3:
        st.metric("Validation accuracy", f"{engine.validation_accuracy:.0%}")
    with c4:
        st.metric("Search index", f"{len(engine.cases):,} records")
    st.caption(f"Retrieval backend: {engine.index_backend}. The persisted index rebuilds only when approved knowledge changes.")

    st.markdown("### Historical examples")
    st.dataframe(engine.case_table(), width="stretch", hide_index=True)
    st.markdown("### Model behavior")
    st.write("The classifier learns patterns from structured case details. SQLite FTS5 first narrows large knowledge libraries to relevant candidates, and persisted TF-IDF vectors rerank those candidates before the answer is written. Low-confidence predictions are routed to a human.")

    feedback_events = load_recommendation_feedback(RECOMMENDATION_FEEDBACK_PATH)
    feedback_history = [record for record in feedback_events if not record.get("event")]
    if str(CURRENT_USER.get("role", "")).lower() in {"administrator", "admin", "expert", "demo"}:
        st.markdown("### Recommendation feedback")
        st.write(
            "Review what employees found helpful, unclear, incomplete, or unsafe. Feedback remains separate from approved knowledge unless an authorized reviewer explicitly promoted a correction."
        )
        helpful_count = sum(item.get("verdict") == "Helpful" for item in feedback_history)
        needs_attention_count = sum(item.get("verdict") != "Helpful" for item in feedback_history)
        approved_correction_count = sum(bool(item.get("expert_approved")) for item in feedback_history)
        feedback_metrics = st.columns(3)
        feedback_metrics[0].metric("Feedback received", len(feedback_history))
        feedback_metrics[1].metric("Needs attention", needs_attention_count)
        feedback_metrics[2].metric("Expert corrections", approved_correction_count)
        st.markdown("#### Learned answer preferences")
        active_profile = learning_profile(feedback_events, "", WORKSPACE_ID)
        for instruction in learning_instructions(active_profile):
            st.write(f"- {instruction}")
        st.caption(f"Learning from {active_profile['feedback_count']} opted-in feedback records. Related negative feedback is applied when a future question shares relevant keywords. This does not retrain model weights.")
        if st.button("Reset learned answer preferences", key="reset_answer_learning"):
            save_recommendation_feedback(RECOMMENDATION_FEEDBACK_PATH, {
                "workspace_id": WORKSPACE_ID, "event": "reset_answer_learning",
                "reviewer": {"user_id": str(CURRENT_USER.get("user_id", ""))},
            })
            st.session_state.pop("assist_answer_record", None)
            st.rerun()
        if feedback_history:
            feedback_rows = []
            for item in reversed(feedback_history):
                ratings = item.get("ratings", {}) if isinstance(item.get("ratings"), dict) else {}
                reviewer = item.get("reviewer", {}) if isinstance(item.get("reviewer"), dict) else {}
                feedback_rows.append(
                    {
                        "Recorded": str(item.get("recorded_at", ""))[:19].replace("T", " "),
                        "Verdict": item.get("verdict", ""),
                        "Relevance": ratings.get("relevance", ""),
                        "Clarity": ratings.get("clarity", ""),
                        "Actionability": ratings.get("actionability", ""),
                        "Issues": "; ".join(str(value) for value in item.get("issue_types", []) if value),
                        "Question": str(item.get("query", ""))[:180],
                        "Reviewer notes": str(item.get("reviewer_notes", ""))[:240],
                        "Suggested correction": str(item.get("suggested_steps", ""))[:300],
                        "Reviewer": reviewer.get("display_name", ""),
                        "Expert approved": bool(item.get("expert_approved")),
                    }
                )
            st.dataframe(pd.DataFrame(feedback_rows), width="stretch", hide_index=True)
        else:
            st.info("No recommendation feedback has been submitted in this company workspace yet.")
        st.caption(f"Helpful submissions: {helpful_count}. Employee email addresses are not displayed in this review table.")
    else:
        st.caption("Recommendation feedback analytics are available to company administrators and expert reviewers.")
