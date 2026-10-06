from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from skillvault.data import ExpertCase, VALID_DECISIONS, cases_from_dataframe, infer_features
from skillvault.engine import SkillVaultEngine
from skillvault.importer import WorkFile, draft_from_completed_work
from skillvault.storage import append_case, load_cases, save_approved_plan, save_cases
from skillvault.workspace import authenticate_user, create_user, load_workspace, save_workspace


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data" / "local" / "blank_workspace"
WORKSPACE_PATH = DATA_DIR / "workspace.json"
USERS_PATH = DATA_DIR / "users.json"
CASES_PATH = DATA_DIR / "approved_expert_decisions.json"
PLANS_PATH = DATA_DIR / "approved_plans.json"
KNOWLEDGE_INDEX_PATH = DATA_DIR / "knowledge_index"

SCENARIO_OPTIONS = [
    "All knowledge",
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
    page_title="SkillVault Clean Start",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --ink:#132238; --muted:#667085; --line:#dfe6ef; --navy:#0e1b2f; --cyan:#00a6a6; --soft:#f5f8fb; }
    .stApp { background: radial-gradient(circle at 80% 0%, rgba(0,166,166,.09), transparent 28rem), var(--soft); color:var(--ink); }
    .block-container { max-width:1180px; padding-top:1.4rem; padding-bottom:4rem; }
    [data-testid="stSidebar"] { background:var(--navy); border-right:1px solid #20304a; }
    [data-testid="stSidebar"] * { color:#e7f0ff !important; }
    [data-testid="stSidebar"] .stRadio label { padding:.45rem .6rem; border-radius:10px; }
    [data-testid="stSidebar"] .stRadio label:hover { background:#192a45; }
    .clean-hero { padding:2rem 2.1rem; border-radius:24px; color:white; background:linear-gradient(120deg,#0e1b2f 0%,#143b56 62%,#008f91 100%); box-shadow:0 18px 45px rgba(19,34,56,.16); margin-bottom:1.5rem; }
    .clean-hero h1 { margin:0; font-size:2.45rem; letter-spacing:-.04em; }
    .clean-hero p { color:#d6edf0; margin:.5rem 0 0; font-size:1.05rem; }
    .eyebrow { color:#73e0de; font-size:.72rem; font-weight:800; letter-spacing:.14em; text-transform:uppercase; margin-bottom:.65rem; }
    .empty-state { padding:2rem; border:1px dashed #aab8c9; border-radius:18px; background:rgba(255,255,255,.72); text-align:center; }
    .setup-card { padding:1.1rem 1.25rem; border:1px solid var(--line); border-radius:16px; background:white; box-shadow:0 8px 24px rgba(30,48,73,.05); }
    .status-dot { display:inline-block; width:9px; height:9px; background:#12b76a; border-radius:50%; margin-right:.35rem; }
    div[data-testid="stTextArea"] textarea, div[data-testid="stTextInput"] input { background:white; border-radius:12px; }
    div[data-testid="stButton"] > button, div[data-testid="stFormSubmitButton"] > button { border-radius:11px; font-weight:700; }
    div[data-testid="stExpander"] { border:1px solid var(--line); border-radius:13px; background:white; }
    </style>
    """,
    unsafe_allow_html=True,
)


def hero(kicker: str, title: str, description: str) -> None:
    st.markdown(
        f'<div class="clean-hero"><div class="eyebrow">{kicker}</div><h1>{title}</h1><p>{description}</p></div>',
        unsafe_allow_html=True,
    )


def empty_knowledge_state(message: str = "This workspace has no approved expert knowledge yet.") -> None:
    st.markdown(
        f'<div class="empty-state"><h3>Start with company knowledge</h3><p>{message}</p><p>Open <strong>Add Company Knowledge</strong> to upload completed work, import a CSV, or record an expert decision.</p></div>',
        unsafe_allow_html=True,
    )


def setup_workspace() -> None:
    hero("Clean-start workspace", "Set up SkillVault", "Describe the organization first. The knowledge base will begin completely empty.")
    left, right = st.columns([1.25, .75])
    with left:
        with st.form("workspace_setup_form"):
            st.markdown("### Organization")
            company_name = st.text_input("Company name", placeholder="Example: Harbor Ridge Manufacturing")
            company_type = st.selectbox(
                "Company type",
                ["Software/SaaS", "Manufacturing", "Healthcare", "Financial services", "Retail/e-commerce", "Professional services", "Education", "Nonprofit", "Government", "Other"],
            )
            industry = st.text_input("Industry or specialty", placeholder="Example: industrial robotics maintenance")
            size = st.selectbox("Company size", ["1–10", "11–50", "51–200", "201–1,000", "1,001–5,000", "5,000+"])
            primary_team = st.text_input("Primary team using SkillVault", placeholder="Example: field service and technical support")
            description = st.text_area(
                "What does the company do?",
                placeholder="Describe its customers, products or services, and the work employees need help with.",
                height=110,
            )
            knowledge_goal = st.text_area(
                "What knowledge should SkillVault preserve?",
                placeholder="Example: troubleshooting decisions, equipment repair methods, escalation rules, client communication, and reporting procedures.",
                height=100,
            )
            sensitivity = st.selectbox("Expected data sensitivity", ["Internal operating knowledge", "May include confidential client information", "May include regulated or highly sensitive information"])

            st.markdown("### Workspace administrator")
            admin_name = st.text_input("Your name")
            admin_email = st.text_input("Work email")
            password = st.text_input("Create password", type="password", help="At least 8 characters. This local prototype stores a salted password hash, not the password itself.")
            confirm_password = st.text_input("Confirm password", type="password")
            submitted = st.form_submit_button("Create clean workspace", type="primary", use_container_width=True)

        if submitted:
            required = {
                "company name": company_name,
                "industry or specialty": industry,
                "primary team": primary_team,
                "company description": description,
                "knowledge goal": knowledge_goal,
                "administrator name": admin_name,
                "work email": admin_email,
            }
            missing = [name for name, value in required.items() if not value.strip()]
            if missing:
                st.error(f"Please complete: {', '.join(missing)}")
            elif "@" not in admin_email:
                st.error("Enter a valid work email.")
            elif len(password) < 8:
                st.error("Use a password with at least 8 characters.")
            elif password != confirm_password:
                st.error("The passwords do not match.")
            else:
                profile = {
                    "company_name": company_name.strip(),
                    "company_type": company_type,
                    "industry": industry.strip(),
                    "company_size": size,
                    "primary_team": primary_team.strip(),
                    "description": description.strip(),
                    "knowledge_goal": knowledge_goal.strip(),
                    "sensitivity": sensitivity,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                try:
                    create_user(USERS_PATH, admin_name, admin_email, password)
                    save_workspace(WORKSPACE_PATH, profile)
                except ValueError as error:
                    st.error(str(error))
                else:
                    st.session_state["blank_user"] = {"display_name": admin_name.strip(), "email": admin_email.strip().lower()}
                    st.success("Workspace created. Your knowledge base currently contains zero decisions.")
                    st.rerun()

    with right:
        st.markdown('<div class="setup-card"><h3>What happens next?</h3><p>1. Sign in to the new workspace.</p><p>2. Upload completed work or prepared expert decisions.</p><p>3. Review every extracted decision.</p><p>4. Ask SkillVault questions grounded only in the approved company knowledge.</p></div>', unsafe_allow_html=True)
        st.info("This is a separate clean-start app. It does not load Northstar Cloud or the existing demo cases.")


def sign_in(profile: dict[str, object]) -> None:
    company_name = str(profile.get("company_name", "SkillVault"))
    hero("Secure workspace", f"Sign in to {company_name}", "Access the company’s approved expert memory and onboarding tools.")
    center = st.columns([1, 1.1, 1])[1]
    with center:
        with st.form("blank_sign_in_form"):
            email = st.text_input("Work email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)
        if submitted:
            user = authenticate_user(USERS_PATH, email, password)
            if user is None:
                st.error("The email or password is incorrect.")
            else:
                st.session_state["blank_user"] = user
                st.rerun()
        st.caption("Local prototype sign-in. A production deployment should use the company’s identity provider and role-based access controls.")


def _case_file_version(path: Path) -> tuple[int, int]:
    if not path.exists():
        return (0, 0)
    stat = path.stat()
    return (stat.st_mtime_ns, stat.st_size)


@st.cache_resource(max_entries=4)
def get_workspace_engine(case_file: str, index_file: str, version: tuple[int, int]) -> SkillVaultEngine:
    """Reuse the model across reruns and rebuild only after knowledge changes."""
    del version
    workspace_engine = SkillVaultEngine(load_cases(Path(case_file)), index_path=Path(index_file))
    workspace_engine.train()
    return workspace_engine


profile = load_workspace(WORKSPACE_PATH)
if not profile:
    setup_workspace()
    st.stop()

if "blank_user" not in st.session_state:
    sign_in(profile)
    st.stop()

user = st.session_state["blank_user"]
engine = get_workspace_engine(str(CASES_PATH), str(KNOWLEDGE_INDEX_PATH), _case_file_version(CASES_PATH))
cases = engine.cases
company_name = str(profile.get("company_name", "Your company"))

with st.sidebar:
    st.markdown("### 🧠 SkillVault")
    st.caption(company_name)
    st.markdown(f'<span class="status-dot"></span>{len(cases)} approved decisions', unsafe_allow_html=True)
    page = st.radio("Workspace", ["Home", "Assist", "Learn", "Add Company Knowledge", "Knowledge Library", "Settings"])
    st.divider()
    st.caption(f"Signed in as {user['display_name']}")
    st.caption(user["email"])
    if st.button("Sign out", use_container_width=True):
        st.session_state.pop("blank_user", None)
        st.session_state.pop("blank_completed_work_draft", None)
        st.rerun()


if page == "Home":
    hero("Company memory workspace", company_name, "Build a private decision library from the work this organization has actually completed.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Approved decisions", len(cases))
    c2.metric("Knowledge sources", len({case.source_file for case in cases}))
    c3.metric("Decision approaches", len({case.label for case in cases}))
    if not cases:
        empty_knowledge_state()
    else:
        st.markdown("### Workspace is ready")
        st.write("SkillVault can now retrieve approved decisions, cite their source files, and tailor the historical process to a new problem.")
        st.dataframe(engine.case_table().tail(5), use_container_width=True, hide_index=True)

elif page == "Assist":
    hero("Evidence-linked assistance", "Ask the company’s expert memory", "Answers use only approved workspace knowledge; weak matches are labeled for human review.")
    if not cases:
        empty_knowledge_state("Assist Mode is locked until at least one expert decision is approved.")
    else:
        query = st.text_area("What do you need help with?", height=150, placeholder="Describe the current task, what changed, who is affected, and the result you need.")
        focus, action = st.columns([1.2, .8])
        with focus:
            scenario = st.selectbox("Knowledge focus", SCENARIO_OPTIONS)
        with action:
            st.write("")
            analyze = st.button("Analyze with company knowledge", type="primary", use_container_width=True)
        if analyze:
            if not query.strip():
                st.warning("Describe the task first.")
            else:
                normalized_scenario = "All scenarios" if scenario == "All knowledge" else scenario
                st.session_state["blank_assist_result"] = engine.predict(query, normalized_scenario)
                st.session_state["blank_assist_query"] = query

        result = st.session_state.get("blank_assist_result")
        if result is not None:
            current_query = st.session_state.get("blank_assist_query", query)
            m1, m2, m3 = st.columns(3)
            m1.metric("Recommended approach", result.label)
            m2.metric("Evidence confidence", f"{result.confidence:.0%}")
            m3.metric("Closest knowledge match", f"{result.knowledge_match:.0%}")
            st.caption(
                f"Indexed retrieval shortlisted {result.candidate_count:,} of {result.indexed_count:,} approved decisions before vector reranking · {result.retrieval_mode}"
            )
            if result.knowledge_match < .35:
                st.warning("The closest company decision is a weak match. Use the answer as a starting point and request expert review.")
            st.markdown("### SkillVault answer")
            st.markdown(engine.compose_response(current_query, result))
            st.markdown("### Sources and similar decisions")
            for similar in result.similar_cases:
                with st.expander(f"{similar.summary} · {similar.similarity:.0%} similarity"):
                    st.write(f"**Approach:** {similar.label}")
                    st.write(f"**Reasoning:** {similar.reasoning}")
                    st.write(f"**Steps:** {similar.instructions or 'Not recorded'}")
                    st.write(f"**Tools:** {similar.software or 'Not recorded'}")
                    st.caption(f"Source: `{similar.source_file}`")
            if st.button("Approve this plan", type="primary"):
                answer = engine.compose_response(current_query, result)
                save_approved_plan(PLANS_PATH, current_query, result.label, result.confidence, result.sources, answer)
                st.success("Approved plan saved to this clean workspace.")

elif page == "Learn":
    hero("Employee training", "Practice with approved decisions", "Compare an employee’s process with the exact playbook captured from company experts.")
    if not cases:
        empty_knowledge_state("Learn Mode needs at least one approved expert decision.")
    else:
        index = st.selectbox("Choose a practice case", range(len(cases)), format_func=lambda i: cases[i].summary)
        case = cases[index]
        st.info(case.summary)
        answer = st.text_area("How would you handle it?", height=170, placeholder="Write your steps, reasoning, verification, and escalation point.")
        if st.button("Compare with the expert", type="primary"):
            if not answer.strip():
                st.warning("Write your approach first.")
            else:
                learning = engine.evaluate_learning(case, answer)
                st.metric("Playbook coverage", f"{learning.score:.0%}")
                st.write(learning.feedback)
                left, right = st.columns(2)
                left.write(f"**Included:** {', '.join(learning.matched_concepts) or 'No key concepts detected'}")
                right.write(f"**Consider adding:** {', '.join(learning.missing_concepts) or 'Main concepts covered'}")
                st.markdown("### Approved expert playbook")
                st.write(case.instructions or case.reasoning)
                st.caption(f"Source: `{case.source_file}`")

elif page == "Add Company Knowledge":
    hero("Knowledge onboarding", "Add the first company decisions", "Nothing becomes model knowledge until an employee reviews and approves it.")
    method = st.radio(
        "Choose an input method",
        ["Upload completed work", "Import prepared CSV", "Record one expert decision"],
        horizontal=True,
        key="blank_knowledge_method",
    )

    if method == "Upload completed work":
        st.markdown("### Turn completed work into a reviewable decision")
        completed_files = st.file_uploader(
            "Upload work evidence",
            type=["txt", "md", "log", "csv", "json", "yaml", "yml", "py", "js", "ts", "html", "css", "sql", "eml", "pdf", "docx", "pptx", "png", "jpg", "jpeg", "webp", "mp4", "mov", "webm"],
            accept_multiple_files=True,
            help="Upload a completed presentation, report, code change, ticket export, email, runbook, screenshot, or recording.",
        )
        notes = st.text_area("Optional employee note or transcript", height=105, placeholder="Explain what was done, why it was chosen, and how the result was verified.")
        if st.button("Create decision draft", type="primary"):
            if not completed_files and not notes.strip():
                st.warning("Upload at least one file or add an employee note.")
            else:
                work_files = [WorkFile(file.name, file.type or "application/octet-stream", file.getvalue()) for file in completed_files]
                st.session_state["blank_completed_work_draft"] = draft_from_completed_work(work_files, notes)

        draft = st.session_state.get("blank_completed_work_draft")
        if draft is not None:
            st.divider()
            st.markdown("### Review the extracted draft")
            for warning in draft.warnings:
                st.warning(warning)
            if draft.evidence_preview:
                with st.expander("Inspect extracted source text"):
                    st.text(draft.evidence_preview)
            with st.form("blank_completed_work_review"):
                summary = st.text_area("Situation", value=draft.summary, height=90)
                decision_options = sorted(VALID_DECISIONS)
                decision = st.selectbox("Expert approach", decision_options, index=decision_options.index(draft.decision))
                reasoning = st.text_area("Reasoning", value=draft.reasoning, height=115)
                instructions = st.text_area("Reusable steps", value=draft.instructions, height=180)
                software = st.text_input("Software and tools", value=draft.software)
                methods = st.text_input("Methods", value=draft.methods)
                outcome = st.text_area("Outcome", value=draft.outcome, height=90)
                source = st.text_input("Source files", value=draft.source_file)
                approved = st.checkbox("I reviewed the evidence and confirm this decision is accurate.")
                submitted = st.form_submit_button("Approve and add to company knowledge", type="primary")
            if submitted:
                required = [summary, reasoning, instructions, outcome, source]
                unresolved = any(marker in value.lower() for value in (reasoning, instructions, outcome) for marker in ("must add", "must record", "replace these draft steps"))
                if any(not value.strip() for value in required):
                    st.error("Complete the situation, reasoning, steps, outcome, and source fields.")
                elif unresolved:
                    st.error("Replace the extraction placeholders with the employee’s actual process.")
                elif not approved:
                    st.warning("An employee must approve the draft first.")
                else:
                    case = ExpertCase(summary, decision, reasoning, infer_features(f"{summary} {reasoning} {instructions}"), source, instructions, software, methods, outcome, draft.media)
                    append_case(CASES_PATH, case)
                    st.session_state.pop("blank_completed_work_draft", None)
                    st.success("The decision was added to this workspace. Assist and Learn modes can now use it.")
                    st.rerun()

    elif method == "Import prepared CSV":
        st.markdown("### Import decisions already organized by the company")
        template = pd.DataFrame([{
            "case": "", "decision": "Follow standard process", "reasoning": "", "instructions": "",
            "source_file": "", "software": "", "methods": "", "outcome": "",
        }])
        st.download_button("Download empty CSV template", template.to_csv(index=False), "skillvault_empty_template.csv", "text/csv")
        uploaded = st.file_uploader("Upload prepared expert decisions", type=["csv"], key="blank_csv_upload")
        if uploaded is not None:
            try:
                frame = pd.read_csv(uploaded)
                edited = st.data_editor(frame, use_container_width=True, num_rows="dynamic")
                imported, errors = cases_from_dataframe(edited)
                if errors:
                    for error in errors[:12]:
                        st.error(error)
                elif st.button("Approve CSV and add knowledge", type="primary"):
                    save_cases(CASES_PATH, load_cases(CASES_PATH) + imported)
                    st.success(f"Added {len(imported)} approved decisions.")
                    st.rerun()
            except Exception as error:
                st.error(f"Could not read this CSV: {error}")

    else:
        st.markdown("### Record one completed expert decision")
        with st.form("blank_manual_decision"):
            summary = st.text_area("What happened?")
            decision = st.selectbox("What approach did the expert use?", sorted(VALID_DECISIONS))
            reasoning = st.text_area("Why was this approach chosen?")
            instructions = st.text_area("What exact steps should someone follow next time?", height=170)
            software = st.text_input("Software and tools used")
            methods = st.text_input("Methods used")
            source = st.text_input("Source file or record name", value="expert_decision.md")
            outcome = st.text_area("What was the outcome?")
            media_files = st.file_uploader("Attach supporting images or videos", type=["png", "jpg", "jpeg", "webp", "mp4", "mov", "webm"], accept_multiple_files=True)
            approved = st.checkbox("I confirm this is an accurate approved decision.")
            submitted = st.form_submit_button("Add decision", type="primary")
        if submitted:
            if any(not value.strip() for value in (summary, reasoning, instructions, source, outcome)):
                st.error("Complete the situation, reasoning, steps, source, and outcome.")
            elif not approved:
                st.warning("Confirm the decision before adding it.")
            else:
                media = [{"name": file.name, "type": file.type or "application/octet-stream", "bytes": file.getvalue()} for file in media_files]
                case = ExpertCase(summary, decision, reasoning, infer_features(f"{summary} {reasoning} {instructions}"), source, instructions, software, methods, outcome, media)
                append_case(CASES_PATH, case)
                st.success("Approved decision added to the clean workspace.")
                st.rerun()

elif page == "Knowledge Library":
    hero("Transparency", "Knowledge Library", "Inspect exactly what this workspace has learned and where each decision came from.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Approved decisions", len(cases))
    c2.metric("Decision approaches", len(engine.labels))
    c3.metric("Validation accuracy", f"{engine.validation_accuracy:.0%}" if engine.uses_classifier else "Building")
    c4.metric("Search index", f"{len(cases):,} records")
    if not cases:
        empty_knowledge_state()
    else:
        st.dataframe(engine.case_table(), use_container_width=True, hide_index=True)
        st.caption(f"Retrieval backend: {engine.index_backend}. The persisted index rebuilds only after approved knowledge changes. A classifier becomes available after the workspace contains at least two different decision approaches.")

else:
    hero("Workspace administration", "Settings", "Review the company profile used to frame this clean-start knowledge workspace.")
    st.markdown("### Company profile")
    profile_table = pd.DataFrame([
        {"Field": "Company name", "Value": profile.get("company_name", "")},
        {"Field": "Company type", "Value": profile.get("company_type", "")},
        {"Field": "Industry", "Value": profile.get("industry", "")},
        {"Field": "Company size", "Value": profile.get("company_size", "")},
        {"Field": "Primary team", "Value": profile.get("primary_team", "")},
        {"Field": "Knowledge goal", "Value": profile.get("knowledge_goal", "")},
        {"Field": "Data sensitivity", "Value": profile.get("sensitivity", "")},
    ])
    st.dataframe(profile_table, use_container_width=True, hide_index=True)
    st.warning("This local prototype is suitable for demonstration and local testing, not production authentication or regulated data. A production version should use SSO, encrypted storage, access levels, audit logs, and retention controls.")
