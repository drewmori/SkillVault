from __future__ import annotations

from pathlib import Path

import streamlit as st
import pandas as pd

from skillvault.data import ExpertCase, VALID_DECISIONS, build_demo_cases, cases_from_dataframe, infer_features
from skillvault.engine import SkillVaultEngine
from skillvault.storage import append_case, load_cases, save_approved_plan, save_cases


APPROVED_CASES_PATH = Path(__file__).with_name("approved_expert_decisions.json")
APPROVED_PLANS_PATH = Path(__file__).with_name("approved_plans.json")
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


@st.cache_resource
def get_engine() -> SkillVaultEngine:
    engine = SkillVaultEngine(build_demo_cases() + load_cases(APPROVED_CASES_PATH))
    engine.train()
    return engine


engine = get_engine()

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

st.markdown(
    '<div class="hero"><div class="hero-kicker">NORTHSTAR CLOUD · INTERNAL KNOWLEDGE SYSTEM</div><h1>🧠 SkillVault</h1><p>An AI apprentice that preserves expert judgment for SaaS teams.</p></div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### ✦ SkillVault")
    st.caption("Northstar Cloud / expert memory")
    mode = st.radio("Choose a mode", ["Assist Mode", "Learn Mode", "Add Expert Decision", "Import Expert Data", "Model Insights"])
    st.caption("Demo company: Northstar Cloud SaaS support and engineering")

if mode == "Assist Mode":
    st.markdown('<div class="section-kicker">Decision support</div>', unsafe_allow_html=True)
    st.subheader("Ask the former expert")
    st.write("Describe a client issue, engineering problem, report, presentation, or company task. SkillVault will use Northstar Cloud's expert knowledge, cite its sources, and flag when escalation is needed.")

    case_text = st.text_area(
        "Task or client issue",
        value="A client reports that file uploads fail for files over 100 MB, while smaller files work. What should I do and what should I tell the client?",
        height=150,
    )
    focus_col, analyze_col = st.columns([1.25, 1])
    with focus_col:
        scenario_filter = st.selectbox("Knowledge focus area", SCENARIO_OPTIONS, help="Prioritize historical decisions from one work area. Choose Other for cases outside the main categories.")
    with analyze_col:
        st.write("")
        analyze_clicked = st.button("Analyze case", type="primary", use_container_width=True)

    if analyze_clicked:
        result = engine.predict(case_text, scenario_filter)
        st.session_state["assist_result"] = result
        st.session_state["assist_query"] = case_text
        st.session_state["assist_scenario"] = scenario_filter
        st.session_state.pop("assist_refined_from", None)

    result = st.session_state.get("assist_result")
    if result is not None:
        case_text = st.session_state.get("assist_query", case_text)
        confidence_class = "confidence-high" if result.confidence >= 0.75 else "confidence-mid"

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Recommended approach", result.label)
        with c2:
            st.markdown(f"<div class='metric-card'><strong>Confidence</strong><br><span class='{confidence_class}'>{result.confidence:.0%}</span></div>", unsafe_allow_html=True)
        with c3:
            st.metric("Review status", "Human review" if result.confidence < 0.75 else "Ready to review")

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
                    st.write(case.summary)
                    st.write(f"**Expert approach:** {case.label}")
                    st.write(f"**Expert instructions:** {case.instructions or case.reasoning}")
                    st.write(f"**Software/tools:** {case.software or 'Not specified'}")
                    st.write(f"**Methods:** {case.methods or 'Not specified'}")
                    if case.outcome:
                        st.write(f"**Outcome:** {case.outcome}")
                    st.caption(f"Source file: `{case.source_file}`")
                    if case.media:
                        st.markdown("**Attached expert media**")
                        for media in case.media:
                            st.caption(str(media["name"]))
                            if str(media["type"]).startswith("image/"):
                                st.image(media["bytes"], use_container_width=True)
                            elif str(media["type"]).startswith("video/"):
                                st.video(media["bytes"])

            st.markdown("### Source files used")
            for source in result.sources:
                st.write(f"📄 `{source}`")

        if result.knowledge_match < 0.35 and "assist_refined_from" not in st.session_state:
            st.warning("SkillVault found a closest historical decision, but the match is weak. Use the original evidence above as a starting point, then add context so the recommendation can be recalculated.")
            clarification = st.text_area(
                "What additional context should SkillVault use?",
                placeholder="For example: who is affected, what changed, what software is involved, what worked before, and what result do you need?",
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
                    st.session_state["assist_result"] = engine.predict(refined_query, scenario_filter)
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
        st.caption("SkillVault keeps the useful expert judgment, identifies what is different today, and rewrites the next steps for this specific request. General suggestions are labeled separately.")
        final_answer = engine.compose_response(case_text, result)
        st.markdown(final_answer)

        st.markdown("### Decision status")
        approve_col, review_col = st.columns(2)
        with approve_col:
            if st.button("Approve final plan", type="primary", key="approve_final_plan"):
                save_approved_plan(APPROVED_PLANS_PATH, case_text, result.label, result.confidence, result.sources, final_answer)
                st.success("Final plan approved and saved to the local decision history.")
        with review_col:
            if st.button("Mark for expert review", key="mark_plan_review"):
                st.info("Marked for expert review. Keep the sources and missing context with the handoff.")

        st.markdown("### Correct the model")
        correction = st.selectbox("If the recommendation is wrong, choose the expert-approved approach", ["No correction", *engine.labels])
        if correction != "No correction" and st.button("Save expert correction"):
            engine.add_feedback(case_text, correction)
            append_case(
                APPROVED_CASES_PATH,
                ExpertCase(
                    case_text,
                    correction,
                    "Added through human expert feedback.",
                    infer_features(case_text),
                    "human_feedback.md",
                    "Human-approved correction from Assist Mode.",
                    "Northstar Cloud Support Console",
                    "Human review; correction feedback",
                    "Saved as an approved correction for future retrieval.",
                ),
            )
            st.success("Correction saved. The model will use it after retraining.")

elif mode == "Learn Mode":
    st.markdown('<div class="section-kicker">Apprenticeship loop</div>', unsafe_allow_html=True)
    st.subheader("Train with the former expert")
    st.write("Solve a realistic task in your own words. SkillVault compares your process with the former expert's playbook and shows what you missed.")

    case_index = st.selectbox("Choose a practice case", range(len(engine.cases)), format_func=lambda i: f"Case {i + 1}: {engine.cases[i].summary[:72]}...")
    case = engine.cases[case_index]
    st.info(case.summary)
    st.caption(f"Knowledge source: `{case.source_file}`")
    st.caption(f"Software/tools: {case.software or 'Not specified'} | Methods: {case.methods or 'Not specified'}")
    answer = st.text_area("How would you solve this? Write your steps and reasoning.", height=180, placeholder="1. First I would...\n2. Then I would...\n3. I would verify...")
    if st.button("Compare with expert", type="primary"):
        if not answer.strip():
            st.warning("Write your approach first so SkillVault can compare it with the expert playbook.")
        else:
            result = engine.evaluate_learning(case, answer)
            st.metric("Playbook coverage", f"{result.score:.0%}")
            st.write(result.feedback)
            left, right = st.columns(2)
            with left:
                st.markdown("### Ideas you included")
                if result.matched_concepts:
                    st.write(", ".join(result.matched_concepts))
                else:
                    st.write("No key concepts detected yet.")
            with right:
                st.markdown("### Ideas to add")
                if result.missing_concepts:
                    st.write(", ".join(result.missing_concepts))
                else:
                    st.write("You covered the main concepts.")
            st.markdown("### Former expert's playbook")
            st.write(case.instructions or case.reasoning)
            st.caption(f"Source file: `{case.source_file}`")
            st.markdown("### Why this matters")
            st.write(engine.teaching_feedback(case))
elif mode == "Add Expert Decision":
    st.markdown('<div class="section-kicker">Knowledge capture</div>', unsafe_allow_html=True)
    st.subheader("Record a completed expert action")
    st.write("Add a decision after the work happened—even days later. SkillVault will only learn it after you confirm that the entry is accurate.")

    with st.form("expert_decision_form"):
        summary = st.text_area("What situation or task happened?", placeholder="A client could not log in after enabling SSO...")
        decision = st.selectbox("What approach did the expert use?", sorted(VALID_DECISIONS))
        reasoning = st.text_area("Why was that approach chosen?", placeholder="The issue affected the whole organization, so it needed identity-specialist review...")
        instructions = st.text_area("What exact steps should someone follow next time?", placeholder="1. Capture the tenant ID...\n2. Check the identity-provider logs...\n3. Escalate with...")
        software = st.text_input("Software and tools used", placeholder="Northstar Cloud Admin; Okta; Jira")
        methods = st.text_input("Methods used", placeholder="SAML trace review; tenant isolation; least-privilege checks")
        source_file = st.text_input("Source file or document name", value="new_expert_decision.md")
        outcome = st.text_area("What was the outcome?", placeholder="Identity engineering fixed the tenant configuration...")
        media_files = st.file_uploader(
            "Attach screenshots, diagrams, or screen recordings",
            type=["png", "jpg", "jpeg", "webp", "mp4", "mov", "webm"],
            accept_multiple_files=True,
            help="Related images and videos will appear in Assist Mode when this expert case is retrieved.",
        )
        approved = st.checkbox("I confirm this is an accurate, approved expert decision.")
        submitted = st.form_submit_button("Save decision and retrain SkillVault", type="primary")

    if submitted:
        missing = [name for name, value in [("situation", summary), ("reasoning", reasoning), ("instructions", instructions), ("source file", source_file), ("outcome", outcome)] if not value.strip()]
        if missing:
            st.error(f"Please complete: {', '.join(missing)}")
        elif not approved:
            st.warning("Please confirm that the expert decision is accurate before adding it.")
        else:
            media = [{"name": file.name, "type": file.type or "application/octet-stream", "bytes": file.getvalue()} for file in media_files]
            new_case = ExpertCase(summary, decision, reasoning, infer_features(summary), source_file, instructions, software, methods, outcome, media)
            engine.add_expert_case(new_case)
            append_case(APPROVED_CASES_PATH, new_case)
            st.success("Expert decision added and SkillVault retrained. Try a similar issue in Assist Mode or Learn Mode.")

elif mode == "Import Expert Data":
    st.markdown('<div class="section-kicker">Knowledge ingestion</div>', unsafe_allow_html=True)
    st.subheader("Import Northstar Cloud expert knowledge")
    st.write("Upload support tickets, runbooks, reports, code guidance, client templates, or case histories. Review the extracted cases, then train SkillVault only after approval.")

    template = pd.DataFrame([
        {"case": "Client upload fails above 100 MB", "decision": "Diagnose and verify", "reasoning": "Reproduce the boundary and inspect client, API, server, and log evidence", "instructions": "Reproduce 99 MB and 101 MB uploads, inspect limits and logs, then document the smallest failing request", "source_file": "upload_incident_2025.md", "software": "Northstar Cloud Console; Northstar Cloud API Gateway; Datadog", "methods": "Boundary testing; log correlation; request tracing", "outcome": "Issue isolated"},
        {"case": "Weekly client health report needs formatting", "decision": "Follow standard process", "reasoning": "Use the standard account summary, risks, metrics, and next actions structure", "instructions": "Start with status, separate wins from risks, include dated metrics, and assign next-action owners", "source_file": "reporting_standards.md", "software": "Northstar Cloud Analytics; Snowflake; Looker", "methods": "Metric reconciliation; timezone checks; cohort analysis", "outcome": "Report approved"},
    ])
    st.download_button("Download CSV template", template.to_csv(index=False), "skillvault_expert_cases_template.csv", "text/csv")
    uploaded = st.file_uploader("Upload expert cases (.csv)", type=["csv"])

    if uploaded is not None:
        try:
            uploaded_frame = pd.read_csv(uploaded)
            st.markdown("### Review uploaded data")
            st.caption("Required columns: case, decision, reasoning. Optional columns: instructions, source_file, software, methods, outcome.")
            edited_frame = st.data_editor(uploaded_frame, use_container_width=True, num_rows="dynamic")
            imported_cases, errors = cases_from_dataframe(edited_frame)
            if errors:
                st.error("Please fix these issues before training:")
                for error in errors[:10]:
                    st.write(f"- {error}")
            elif st.button("Approve data and train SkillVault", type="primary"):
                engine.add_expert_cases(imported_cases)
                save_cases(APPROVED_CASES_PATH, load_cases(APPROVED_CASES_PATH) + imported_cases)
                st.success(f"SkillVault retrained on {len(imported_cases)} approved expert cases.")
                st.info("Go to Assist Mode or Learn Mode to use the updated model.")
        except Exception as error:
            st.error(f"Could not read this CSV: {error}")

else:
    st.markdown('<div class="section-kicker">Transparency layer</div>', unsafe_allow_html=True)
    st.subheader("Model Insights")
    st.write("A transparent view of what SkillVault has learned from the expert's historical cases.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Expert examples", len(engine.cases))
    with c2:
        st.metric("Decision classes", len(engine.labels))
    with c3:
        st.metric("Validation accuracy", f"{engine.validation_accuracy:.0%}")

    st.markdown("### Historical examples")
    st.dataframe(engine.case_table(), use_container_width=True, hide_index=True)
    st.markdown("### Model behavior")
    st.write("The classifier learns patterns from structured case details while similar-case retrieval keeps recommendations grounded in real examples. Low-confidence predictions are routed to a human.")
