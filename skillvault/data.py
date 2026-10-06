from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd


LIFECYCLE_STATUSES = ("Current", "Outdated", "Replaced")
DEMO_APPROVAL_DATE = "2026-07-13"
DEMO_EXPIRATION_DATE = "2027-07-13"


@dataclass
class ExpertCase:
    summary: str
    label: str
    reasoning: str
    features: dict[str, float]
    source_file: str = "synthetic_expert_cases.csv"
    instructions: str = ""
    software: str = ""
    methods: str = ""
    outcome: str = ""
    media: list[dict[str, object]] = field(default_factory=list)
    expert_owner: str = "Northstar Cloud Knowledge Council"
    department: str = "Cross-functional"
    approval_date: str = DEMO_APPROVAL_DATE
    last_reviewed_date: str = DEMO_APPROVAL_DATE
    expiration_date: str = DEMO_EXPIRATION_DATE
    lifecycle_status: str = "Current"
    replaces_source: str = ""
    goal: str = ""
    chosen_approach: str = ""
    alternatives_considered: str = ""
    constraints: str = ""
    reusable_rule: str = ""
    exceptions: str = ""
    capture_evidence: dict[str, object] = field(default_factory=dict)


REQUIRED_COLUMNS = {"case", "decision", "reasoning"}
VALID_DECISIONS = {
    "Follow standard process",
    "Diagnose and verify",
    "Build in small steps",
    "Write and test",
    "Escalate for review",
}


def _clean_optional(value: object, default: str = "") -> str:
    if value is None or pd.isna(value):
        return default
    cleaned = str(value).strip()
    return default if not cleaned or cleaned.lower() == "nan" else cleaned


def _iso_date(value: object, default: date) -> str:
    cleaned = _clean_optional(value)
    if not cleaned:
        return default.isoformat()
    try:
        return date.fromisoformat(cleaned[:10]).isoformat()
    except ValueError:
        return default.isoformat()


def normalize_lifecycle_status(value: object) -> str:
    cleaned = _clean_optional(value, "Current").lower()
    return next((status for status in LIFECYCLE_STATUSES if status.lower() == cleaned), "Current")


def effective_lifecycle_status(case: ExpertCase, today: date | None = None) -> str:
    """Return the stored status, automatically treating expired knowledge as outdated."""
    status = normalize_lifecycle_status(case.lifecycle_status)
    if status in {"Outdated", "Replaced"}:
        return status
    try:
        expires = date.fromisoformat(case.expiration_date[:10])
    except (TypeError, ValueError):
        return "Outdated"
    return "Outdated" if expires < (today or date.today()) else "Current"


def decision_is_retrievable(case: ExpertCase, today: date | None = None) -> bool:
    return effective_lifecycle_status(case, today) == "Current"


def cases_from_dataframe(frame: pd.DataFrame) -> tuple[list[ExpertCase], list[str]]:
    """Convert an approved upload into model-ready cases."""
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        return [], [f"Missing required column: {column}" for column in missing]

    cases: list[ExpertCase] = []
    errors: list[str] = []
    today = date.today()
    for row_number, row in frame.iterrows():
        summary = str(row["case"]).strip()
        label = str(row["decision"]).strip()
        reasoning = str(row["reasoning"]).strip()
        if not summary or summary.lower() == "nan":
            errors.append(f"Row {row_number + 2}: case is empty")
        if label not in VALID_DECISIONS:
            errors.append(f"Row {row_number + 2}: decision must be one of {sorted(VALID_DECISIONS)}")
        if not reasoning or reasoning.lower() == "nan":
            errors.append(f"Row {row_number + 2}: reasoning is empty")
        source_file = _clean_optional(row.get("source_file"), "uploaded_expert_cases.csv")
        instructions = _clean_optional(row.get("instructions"), reasoning)
        software = _clean_optional(row.get("software"))
        methods = _clean_optional(row.get("methods"))
        outcome = _clean_optional(row.get("outcome"))
        expert_owner = _clean_optional(row.get("expert_owner"), "Imported knowledge owner")
        department = _clean_optional(row.get("department"), "Unassigned")
        approval_date = _iso_date(row.get("approval_date"), today)
        try:
            approved_on = date.fromisoformat(approval_date)
        except ValueError:
            approved_on = today
        last_reviewed_date = _iso_date(row.get("last_reviewed_date"), approved_on)
        expiration_date = _iso_date(row.get("expiration_date"), approved_on + timedelta(days=365))
        lifecycle_status = normalize_lifecycle_status(row.get("lifecycle_status"))
        replaces_source = _clean_optional(row.get("replaces_source"))
        goal = _clean_optional(row.get("goal"))
        chosen_approach = _clean_optional(row.get("chosen_approach"))
        alternatives_considered = _clean_optional(row.get("alternatives_considered"))
        constraints = _clean_optional(row.get("constraints"))
        reusable_rule = _clean_optional(row.get("reusable_rule"))
        exceptions = _clean_optional(row.get("exceptions"))
        cases.append(
            ExpertCase(
                summary,
                label,
                reasoning,
                infer_features(summary),
                source_file,
                instructions,
                software,
                methods,
                outcome,
                expert_owner=expert_owner,
                department=department,
                approval_date=approval_date,
                last_reviewed_date=last_reviewed_date,
                expiration_date=expiration_date,
                lifecycle_status=lifecycle_status,
                replaces_source=replaces_source,
                goal=goal,
                chosen_approach=chosen_approach,
                alternatives_considered=alternatives_considered,
                constraints=constraints,
                reusable_rule=reusable_rule,
                exceptions=exceptions,
            )
        )
    return cases, errors


def infer_features(text: str) -> dict[str, float]:
    lowered = text.lower()
    return {
        "client_issue": float(any(word in lowered for word in ("client", "customer", "user", "reporting"))),
        "bug": float(any(word in lowered for word in ("bug", "error", "crash", "failure", "broken"))),
        "code": float(any(word in lowered for word in ("code", "python", "api", "function", "developer"))),
        "presentation": float(any(word in lowered for word in ("presentation", "powerpoint", "investor", "slide", "pitch", "deck"))),
        "report": float(any(word in lowered for word in ("report", "metric", "dashboard", "analysis"))),
        "urgent": float(any(word in lowered for word in ("urgent", "outage", "security", "blocked", "many customers"))),
    }


def build_demo_cases() -> list[ExpertCase]:
    """Synthetic Northstar Cloud cases used for the competition demo."""
    cases = [
        ExpertCase(
            "A client reports that file uploads fail for files over 100 MB, while smaller files work.",
            "Diagnose and verify",
            "Reproduce the issue with files just below and above 100 MB, then compare client limits, API limits, server timeouts, and upload logs.",
            {"client_issue": 1, "bug": 1, "code": 0, "presentation": 0, "report": 0, "urgent": 0},
            "upload_incident_2025.md",
            "1. Reproduce with a 99 MB and a 101 MB file.\n2. Check browser, API gateway, and server size limits.\n3. Inspect timeout and upload logs.\n4. Record the smallest failing request before proposing a fix.",
        ),
        ExpertCase(
            "A client says dashboard numbers are different from the CSV export and wants an answer today.",
            "Diagnose and verify",
            "Verify the date range, timezone, filters, and aggregation logic before promising that one number is correct.",
            {"client_issue": 1, "bug": 1, "code": 0, "presentation": 0, "report": 1, "urgent": 1},
            "analytics_support_playbook.md",
            "1. Capture the exact dashboard filters and export settings.\n2. Compare timezone and date boundaries.\n3. Reconcile one account manually.\n4. Share findings with the client before changing data.",
        ),
        ExpertCase(
            "A client cannot log in after enabling single sign-on for their organization.",
            "Escalate for review",
            "SSO configuration can affect an entire organization. Collect the identity-provider error, tenant ID, and timestamp, then escalate to the identity specialist.",
            {"client_issue": 1, "bug": 1, "code": 0, "presentation": 0, "report": 0, "urgent": 1},
            "identity_escalation_rules.md",
            "1. Do not disable SSO without approval.\n2. Capture tenant ID, IdP, error text, and timestamp.\n3. Check whether all users or one user is affected.\n4. Escalate with a complete handoff.",
        ),
        ExpertCase(
            "Several customers report slow requests immediately after a new API deployment.",
            "Escalate for review",
            "A multi-customer regression after deployment may be an incident. Check monitoring and rollback criteria, then notify the incident owner.",
            {"client_issue": 1, "bug": 1, "code": 1, "presentation": 0, "report": 0, "urgent": 1},
            "incident_response_runbook.md",
            "1. Confirm the timing against the deployment.\n2. Check latency, error rate, and affected endpoints.\n3. Notify the incident owner.\n4. Recommend rollback only through the approved incident process.",
        ),
        ExpertCase(
            "A client asks whether a feature can be enabled for only their team.",
            "Follow standard process",
            "Check feature-availability rules and tenant configuration before promising a custom behavior.",
            {"client_issue": 1, "bug": 0, "code": 0, "presentation": 0, "report": 0, "urgent": 0},
            "client_configuration_guide.md",
            "1. Identify the tenant and requested team scope.\n2. Check whether the feature supports team-level configuration.\n3. Confirm permissions and rollout status.\n4. Reply with the supported option and limitations.",
        ),
        ExpertCase(
            "A support engineer needs to write a response to a client whose issue is still being investigated.",
            "Follow standard process",
            "A good response acknowledges the impact, states what is known, avoids unsupported promises, and gives the next update time.",
            {"client_issue": 1, "bug": 0, "code": 0, "presentation": 0, "report": 0, "urgent": 0},
            "customer_response_templates.md",
            "1. Acknowledge the specific impact.\n2. State the confirmed facts only.\n3. Explain the next investigation step.\n4. Give a concrete update time and owner.",
        ),
        ExpertCase(
            "A client is angry because the same issue has happened three times this month.",
            "Escalate for review",
            "Repeated failures require account-level visibility. Summarize the pattern, impact, and prior fixes for the account owner and engineering lead.",
            {"client_issue": 1, "bug": 1, "code": 0, "presentation": 0, "report": 0, "urgent": 1},
            "customer_escalation_policy.md",
            "1. List each occurrence with dates and impact.\n2. Separate temporary workarounds from permanent fixes.\n3. Notify the account owner.\n4. Create a follow-up plan instead of sending another generic apology.",
        ),
        ExpertCase(
            "A Python service returns a TypeError only when a client sends an optional field as null.",
            "Write and test",
            "Reproduce the null input, define the expected API behavior, add a regression test, and implement the smallest safe fix.",
            {"client_issue": 1, "bug": 1, "code": 1, "presentation": 0, "report": 0, "urgent": 0},
            "engineering_runbook.md",
            "1. Add the smallest failing request to a test.\n2. Decide whether null means missing, empty, or invalid.\n3. Implement the narrowest compatible fix.\n4. Test null, missing, and valid values before opening a handoff.",
        ),
        ExpertCase(
            "An engineer needs to hand an unresolved client bug to the product team.",
            "Follow standard process",
            "A useful handoff contains reproduction steps, expected and actual behavior, environment details, evidence, impact, and a proposed priority.",
            {"client_issue": 1, "bug": 1, "code": 1, "presentation": 0, "report": 0, "urgent": 0},
            "engineering_handoff_template.md",
            "1. Write exact reproduction steps.\n2. Include expected versus actual behavior.\n3. Attach logs or screenshots without secrets.\n4. State affected customers and business impact.\n5. Ask one clear question of engineering.",
        ),
        ExpertCase(
            "A report needs to explain weekly retention to executives who do not work with data.",
            "Build in small steps",
            "Lead with the business message, define the metric in plain language, show the trend, and separate evidence from recommendations.",
            {"client_issue": 0, "bug": 0, "code": 0, "presentation": 1, "report": 1, "urgent": 0},
            "reporting_standards.md",
            "1. State the one-sentence takeaway first.\n2. Define retention and the comparison period.\n3. Show one readable trend visual.\n4. Explain likely causes separately from confirmed evidence.\n5. End with a decision or next step.",
        ),
        ExpertCase(
            "A product pitch has ten features but the audience only needs to understand the core value.",
            "Build in small steps",
            "Start with the customer problem, show the smallest compelling workflow, and move extra features into backup slides.",
            {"client_issue": 0, "bug": 0, "code": 0, "presentation": 1, "report": 0, "urgent": 0},
            "presentation_playbook.md",
            "1. Name the audience's problem.\n2. Show the before-and-after workflow.\n3. Demonstrate one core feature.\n4. Quantify the benefit if evidence exists.\n5. Put the feature list in an appendix.",
        ),
        ExpertCase(
            "A presentation has dense paragraphs, inconsistent spacing, and no clear conclusion.",
            "Follow standard process",
            "Use one message per slide, a consistent visual system, and a final slide that states the decision or ask.",
            {"client_issue": 0, "bug": 0, "code": 0, "presentation": 1, "report": 0, "urgent": 0},
            "presentation_playbook.md",
            "1. Rewrite each slide title as a conclusion.\n2. Remove repeated text and move detail to notes.\n3. Apply one layout, type scale, and spacing system.\n4. End with a clear ask.",
        ),
        ExpertCase(
            "A manager wants a recommendation, but the available report has contradictory numbers.",
            "Escalate for review",
            "Do not choose the number that supports the preferred story. Document the conflict, identify the source systems, and request data-owner review.",
            {"client_issue": 0, "bug": 1, "code": 0, "presentation": 0, "report": 1, "urgent": 0},
            "reporting_standards.md",
            "1. Freeze the conflicting inputs.\n2. Record definitions, filters, and timestamps.\n3. Identify the owner of each source.\n4. Escalate with a focused question and no invented conclusion.",
        ),
        ExpertCase(
            "A new support engineer is unsure how to prioritize a bug affecting one large client versus a minor issue affecting many users.",
            "Diagnose and verify",
            "Compare customer impact, number of users, severity, workaround availability, contractual commitments, and security risk before assigning priority.",
            {"client_issue": 1, "bug": 1, "code": 0, "presentation": 0, "report": 0, "urgent": 1},
            "support_triage_playbook.md",
            "1. Quantify affected users and revenue impact.\n2. Check severity and security implications.\n3. Record workaround availability.\n4. Check contractual commitments.\n5. Assign priority with evidence, not volume alone.",
        ),
        ExpertCase(
            "A client asks for a custom feature that is not on the roadmap.",
            "Follow standard process",
            "Clarify the underlying outcome, check existing configuration options, and capture the request without promising delivery.",
            {"client_issue": 1, "bug": 0, "code": 0, "presentation": 0, "report": 0, "urgent": 0},
            "customer_response_templates.md",
            "1. Ask what outcome the client needs.\n2. Check supported workarounds.\n3. Capture the request and business impact.\n4. State that roadmap decisions require product review.",
        ),
        ExpertCase(
            "A user asks how to format a weekly client health report.",
            "Follow standard process",
            "Use the standard structure: account summary, wins, risks, usage metrics, open issues, next actions, and owner.",
            {"client_issue": 1, "bug": 0, "code": 0, "presentation": 0, "report": 1, "urgent": 0},
            "reporting_standards.md",
            "1. Start with account status and the one-line summary.\n2. Separate wins from risks.\n3. Include usage and support metrics with dates.\n4. List next actions, owners, and deadlines.",
        ),
        ExpertCase(
            "A support engineer asks how to investigate an issue that is not covered in the company runbooks.",
            "Escalate for review",
            "Use safe general debugging steps, clearly mark assumptions, preserve evidence, and escalate if the issue affects security, data integrity, or multiple customers.",
            {"client_issue": 1, "bug": 1, "code": 1, "presentation": 0, "report": 0, "urgent": 0},
            "support_triage_playbook.md",
            "1. Reproduce without changing production data.\n2. Collect timestamps, IDs, logs, and environment details.\n3. Try a reversible low-risk check.\n4. Escalate if impact, security, or data integrity is unclear.",
        ),
        ExpertCase(
            "A client asks for a copy of all data stored about their users.",
            "Escalate for review",
            "Data-export requests can have privacy and contractual implications. Verify the requester and route the request through the approved privacy process.",
            {"client_issue": 1, "bug": 0, "code": 0, "presentation": 0, "report": 0, "urgent": 1},
            "privacy_request_runbook.md",
            "1. Verify the requester and tenant ownership.\n2. Do not export data directly from production.\n3. Open the approved privacy request.\n4. Ask the privacy owner to confirm scope and delivery method.",
        ),
        ExpertCase(
            "A client says they were charged twice for the same subscription.",
            "Diagnose and verify",
            "Compare the billing provider events, invoice IDs, and subscription state before issuing a correction.",
            {"client_issue": 1, "bug": 1, "code": 0, "presentation": 0, "report": 1, "urgent": 0},
            "billing_support_playbook.md",
            "1. Collect tenant and invoice IDs.\n2. Compare Stripe events with the Northstar Cloud billing ledger.\n3. Check whether one charge is an authorization hold.\n4. Escalate refund approval when the ledger confirms duplication.",
        ),
        ExpertCase(
            "A client requests an urgent production change directly in a support chat.",
            "Follow standard process",
            "Production changes must use the change-management path even when the request is urgent.",
            {"client_issue": 1, "bug": 0, "code": 1, "presentation": 0, "report": 0, "urgent": 1},
            "change_management_policy.md",
            "1. Capture the requested outcome and impact.\n2. Create the change ticket in Jira.\n3. Record rollback steps and an owner.\n4. Get the required approval before changing production.",
        ),
        ExpertCase(
            "A release needs customer-facing notes for a feature that changes the API response format.",
            "Write and test",
            "API behavior changes need an accurate example, migration guidance, and a compatibility warning.",
            {"client_issue": 0, "bug": 0, "code": 1, "presentation": 0, "report": 0, "urgent": 0},
            "release_communications_guide.md",
            "1. Compare old and new response examples.\n2. Identify affected versions and clients.\n3. Write a migration example.\n4. Have engineering verify the note before publishing.",
        ),
        ExpertCase(
            "A sales teammate needs to pitch Northstar Cloud to a prospect concerned about security.",
            "Follow standard process",
            "Lead with verified security capabilities and never promise certifications or controls that are not documented.",
            {"client_issue": 1, "bug": 0, "code": 0, "presentation": 1, "report": 0, "urgent": 0},
            "security_pitch_guide.md",
            "1. Ask which security concern drives the decision.\n2. Use only approved security language.\n3. Link to the current trust-center material.\n4. Route custom security questionnaires to the security owner.",
        ),
        ExpertCase(
            "A new employee needs to learn the support process for a high-severity client incident.",
            "Build in small steps",
            "Teach the incident workflow through a scenario: detect, confirm impact, communicate, mitigate, document, and review.",
            {"client_issue": 1, "bug": 1, "code": 0, "presentation": 0, "report": 0, "urgent": 1},
            "support_onboarding_curriculum.md",
            "1. Practice with a simulated incident.\n2. Identify impact and incident owner.\n3. Draft the first client update.\n4. Perform a safe mitigation.\n5. Write the timeline and retrospective.",
        ),
        ExpertCase(
            "A weekly team report has tasks but no owners or deadlines.",
            "Follow standard process",
            "Every reported action should have one accountable owner, a due date, and a measurable completion condition.",
            {"client_issue": 0, "bug": 0, "code": 0, "presentation": 0, "report": 1, "urgent": 0},
            "team_reporting_standards.md",
            "1. Rewrite each task as an outcome.\n2. Assign one owner.\n3. Add a due date.\n4. Define what evidence marks it complete.",
        ),
        ExpertCase(
            "A client wants a meeting to discuss an issue that already has a written resolution.",
            "Follow standard process",
            "Send a concise written summary first, then schedule a meeting when the issue requires discussion, tradeoffs, or relationship repair.",
            {"client_issue": 1, "bug": 0, "code": 0, "presentation": 0, "report": 0, "urgent": 0},
            "customer_communication_guide.md",
            "1. Summarize the issue and resolution.\n2. State the next action and owner.\n3. Offer a meeting if there are open decisions or unresolved impact.\n4. Avoid meetings that only repeat the ticket.",
        ),
        ExpertCase(
            "An engineer wants to query production data directly to investigate a client issue.",
            "Escalate for review",
            "Production access can expose customer data and should use approved read-only tools or an authorized data owner.",
            {"client_issue": 1, "bug": 1, "code": 1, "presentation": 0, "report": 0, "urgent": 0},
            "production_access_policy.md",
            "1. Use a sanitized reproduction first.\n2. Request read-only access through the approved channel.\n3. Minimize identifiers and avoid copying customer data.\n4. Record the query and reviewer.",
        ),
        ExpertCase(
            "A product manager needs to turn customer feedback into a product proposal.",
            "Build in small steps",
            "Group feedback by underlying problem, quantify frequency and impact, then propose a small testable change before a large build.",
            {"client_issue": 1, "bug": 0, "code": 0, "presentation": 1, "report": 1, "urgent": 0},
            "product_feedback_framework.md",
            "1. Remove duplicate requests.\n2. Separate requested features from underlying jobs.\n3. Quantify impacted accounts.\n4. Propose a small experiment with a success metric.",
        ),
        ExpertCase(
            "A client says a workaround fixed the issue but wants to know whether it will happen again.",
            "Diagnose and verify",
            "Do not claim the issue is permanently fixed until the root cause, scope, and monitoring plan are understood.",
            {"client_issue": 1, "bug": 1, "code": 0, "presentation": 0, "report": 0, "urgent": 0},
            "incident_response_runbook.md",
            "1. Explain that the workaround reduced the symptom.\n2. State what is known about root cause.\n3. Describe the permanent-fix plan or remaining uncertainty.\n4. Set the next update time.",
        ),
        ExpertCase(
            "A team wants to automate a repetitive support task that has no documented procedure.",
            "Escalate for review",
            "Automating an undocumented process can scale mistakes. Document the current process and get an owner to approve the intended behavior first.",
            {"client_issue": 1, "bug": 0, "code": 1, "presentation": 0, "report": 0, "urgent": 0},
            "automation_review_policy.md",
            "1. Observe and document the current workflow.\n2. Identify exceptions and human approvals.\n3. Define success and failure states.\n4. Test in a sandbox before proposing automation.",
        ),
        ExpertCase(
            "A Python worker processes duplicate webhook events and creates two identical customer records.",
            "Write and test",
            "Webhook delivery is at-least-once, so the worker needs an idempotency key and a database constraint rather than a timing-based duplicate check.",
            {"client_issue": 1, "bug": 1, "code": 1, "presentation": 0, "report": 0, "urgent": 0},
            "webhook_reliability_runbook.md",
            "1. Save a failing test with the same event delivered twice.\n2. Use the provider event ID as an idempotency key.\n3. Add a unique constraint or atomic upsert.\n4. Test retries, out-of-order events, and concurrent workers.\n5. Check existing duplicates before deploying the fix.",
            "Python; pytest; PostgreSQL; Northstar Cloud Events API",
            "Idempotent consumers; atomic upsert; concurrency testing",
            "Duplicate creation stopped in staging and a cleanup plan was opened for existing records.",
        ),
        ExpertCase(
            "A TypeScript API returns HTTP 200 with an error object when a required field is missing.",
            "Write and test",
            "The response contract is ambiguous and can cause clients to treat failures as success. Define the error schema and preserve the existing valid response shape.",
            {"client_issue": 1, "bug": 1, "code": 1, "presentation": 0, "report": 0, "urgent": 0},
            "api_contract_standards.md",
            "1. Capture the current request and response.\n2. Add tests for missing, null, empty, and valid values.\n3. Return the approved 4xx status and error code.\n4. Update the OpenAPI example.\n5. Ask a consuming team to verify compatibility before release.",
            "TypeScript; Jest; OpenAPI; Postman",
            "Contract testing; schema validation; backward-compatibility review",
            "The endpoint returned consistent status codes and the client SDK handled the documented error.",
        ),
        ExpertCase(
            "A database query became slow after a report added a filter on an unindexed timestamp column.",
            "Diagnose and verify",
            "Measure the query plan before adding an index; confirm the filter is the cause and check whether the index will create unacceptable write cost.",
            {"client_issue": 0, "bug": 1, "code": 1, "presentation": 0, "report": 1, "urgent": 0},
            "database_performance_runbook.md",
            "1. Capture query duration and affected tenants.\n2. Run EXPLAIN without changing data.\n3. Compare the query before and after the new filter.\n4. Test an index in a staging-sized copy.\n5. Review storage and write overhead before production approval.",
            "PostgreSQL; Snowflake; Datadog",
            "Query-plan comparison; representative-load testing; reversible migration",
            "The team confirmed the scan, tested the index safely, and scheduled the change during the maintenance window.",
        ),
        ExpertCase(
            "A production service is using much more memory after a feature flag was enabled for 10 percent of tenants.",
            "Escalate for review",
            "A controlled rollout is still a production incident when resource use rises unexpectedly. Preserve the rollout data and involve the incident owner before changing flags.",
            {"client_issue": 1, "bug": 1, "code": 1, "presentation": 0, "report": 0, "urgent": 1},
            "incident_response_runbook.md",
            "1. Record the flag, cohort, deployment, and first alert time.\n2. Compare memory by flagged and unflagged cohort.\n3. Notify the incident owner and on-call engineer.\n4. Follow the approved mitigation or flag rollback.\n5. Keep a timeline of every change and observation.",
            "LaunchDarkly; Datadog; GitHub Actions",
            "Cohort comparison; incident timeline; controlled mitigation",
            "The rollout was paused through the incident process and the team opened a root-cause investigation.",
        ),
        ExpertCase(
            "A client sees a blank page only in Safari while Chrome works normally.",
            "Diagnose and verify",
            "Browser-specific failures need a minimal comparison across browser version, console errors, network requests, cached assets, and account configuration.",
            {"client_issue": 1, "bug": 1, "code": 1, "presentation": 0, "report": 0, "urgent": 0},
            "browser_issue_playbook.md",
            "1. Capture Safari version and operating system.\n2. Reproduce in a private window.\n3. Save console and network errors with secrets removed.\n4. Compare the failing request with Chrome.\n5. Test the latest supported browser before escalating.",
            "Safari Web Inspector; Chrome DevTools; Northstar Cloud Console",
            "Cross-browser reproduction; console inspection; cache isolation",
            "The team isolated an unsupported browser API and prepared a compatibility fix with a temporary browser recommendation.",
        ),
        ExpertCase(
            "A client reports that notifications arrive hours late, but the notification job shows as successful.",
            "Diagnose and verify",
            "A successful enqueue is not proof of delivery. Trace the message from scheduling through queue, provider response, and client receipt.",
            {"client_issue": 1, "bug": 1, "code": 1, "presentation": 0, "report": 0, "urgent": 0},
            "notification_delivery_runbook.md",
            "1. Collect tenant ID, notification ID, and intended time.\n2. Trace enqueue, queue delay, provider response, and delivery receipt.\n3. Compare delayed messages by channel and region.\n4. Check provider throttling and retry counts.\n5. Communicate whether the delay is internal or provider-side.",
            "Datadog; SQS; SendGrid; Northstar Cloud Events API",
            "Distributed tracing; queue-age analysis; provider-response correlation",
            "The investigation separated queue delay from provider throttling and gave the client an evidence-based update.",
        ),
        ExpertCase(
            "An engineer wants to hotfix a production bug by editing a container manually.",
            "Escalate for review",
            "Manual edits are not reproducible and can disappear on restart. Use the version-controlled release path unless the incident commander approves a documented emergency action.",
            {"client_issue": 0, "bug": 1, "code": 1, "presentation": 0, "report": 0, "urgent": 1},
            "change_management_policy.md",
            "1. Capture the impact and current version.\n2. Create a reproducible branch or patch.\n3. Define rollback and verification checks.\n4. Get incident-owner approval for an emergency release.\n5. Record the exact commit and post-change evidence.",
            "GitHub; Docker; GitHub Actions; Datadog",
            "Version-controlled change; rollback planning; post-deploy verification",
            "The fix was shipped through a traceable emergency release instead of an undocumented container edit.",
        ),
        ExpertCase(
            "A code review contains a large refactor mixed with a small production bug fix.",
            "Follow standard process",
            "Separating unrelated changes makes review and rollback safer. Keep the bug fix narrow and open a separate refactor change.",
            {"client_issue": 0, "bug": 1, "code": 1, "presentation": 0, "report": 0, "urgent": 0},
            "engineering_review_standards.md",
            "1. Identify files required for the bug fix.\n2. Move unrelated formatting and refactoring to a separate branch.\n3. Add a focused regression test.\n4. Explain risk, test results, and rollback in the pull request.\n5. Request review from the owning team.",
            "GitHub; Python; pytest",
            "Small diffs; regression testing; reviewer ownership",
            "The focused fix was reviewed quickly and the refactor received separate design feedback.",
        ),
        ExpertCase(
            "A client asks support to delete records immediately without explaining which records or why.",
            "Escalate for review",
            "Deletion is irreversible and may conflict with retention, legal hold, billing, or audit requirements. Clarify scope and route it to the approved data process.",
            {"client_issue": 1, "bug": 0, "code": 0, "presentation": 0, "report": 0, "urgent": 1},
            "privacy_request_runbook.md",
            "1. Ask for tenant, record type, identifiers, and requested deadline.\n2. Do not delete from production manually.\n3. Check retention and legal-hold status with the data owner.\n4. Open the approved deletion request.\n5. Require a completion record and verification.",
            "Northstar Cloud Admin; Jira; Snowflake",
            "Scope confirmation; retention review; least-privilege execution",
            "The request was scoped and approved before any irreversible action occurred.",
        ),
        ExpertCase(
            "A client says a monthly usage report is lower than expected after a timezone change.",
            "Diagnose and verify",
            "Timezone boundaries can move events between reporting periods. Reconcile a small sample using UTC timestamps before changing the report logic.",
            {"client_issue": 1, "bug": 1, "code": 0, "presentation": 0, "report": 1, "urgent": 0},
            "reporting_standards.md",
            "1. Record the report timezone and comparison period.\n2. Select a sample of events near midnight.\n3. Compare stored UTC time with displayed local time.\n4. Recalculate the sample independently.\n5. Explain the boundary effect before changing production reporting.",
            "Northstar Cloud Analytics; Snowflake; Looker",
            "UTC reconciliation; boundary sampling; independent recalculation",
            "The report was corrected with a documented timezone definition and the client received a clear explanation.",
        ),
        ExpertCase(
            "An executive asks for a single KPI but the team has three slightly different definitions of active customer.",
            "Escalate for review",
            "A polished chart cannot resolve an undefined metric. Get a data owner and business owner to approve one definition before publishing the KPI.",
            {"client_issue": 0, "bug": 1, "code": 0, "presentation": 1, "report": 1, "urgent": 0},
            "reporting_standards.md",
            "1. List each current definition and source.\n2. Show how the result changes under each definition.\n3. Ask the business owner which decision the KPI supports.\n4. Document the approved definition, date range, and exclusions.\n5. Add the definition to the report itself.",
            "Snowflake; Looker; PowerPoint",
            "Metric contract; decision-oriented reporting; definition governance",
            "Leadership approved one KPI definition and the report stopped presenting conflicting totals.",
        ),
        ExpertCase(
            "A client response needs to explain a temporary workaround without sounding like the issue is permanently fixed.",
            "Follow standard process",
            "Separate what the workaround changes from what remains unknown, and give the client a specific next update instead of implying resolution.",
            {"client_issue": 1, "bug": 1, "code": 0, "presentation": 0, "report": 0, "urgent": 0},
            "customer_response_templates.md",
            "1. Name the symptom and confirmed impact.\n2. Explain the temporary workaround and its limitation.\n3. Say whether root cause is confirmed.\n4. Give the next update time and owner.\n5. Avoid words like fixed or resolved until verification is complete.",
            "Northstar Cloud Support Console; Jira",
            "Expectation setting; fact-versus-assumption separation; time-bound updates",
            "The client understood the workaround while the investigation remained open.",
        ),
        ExpertCase(
            "A support engineer receives a log file containing an API key in a client attachment.",
            "Escalate for review",
            "Potential credential exposure is a security event. Do not paste the key into more tickets; preserve minimal evidence and notify security through the approved path.",
            {"client_issue": 1, "bug": 0, "code": 1, "presentation": 0, "report": 0, "urgent": 1},
            "security_incident_runbook.md",
            "1. Restrict access to the attachment.\n2. Do not copy or repeat the secret.\n3. Record the source, time, tenant, and affected system without exposing the key.\n4. Notify security and the incident owner.\n5. Follow approved rotation and client-notification steps.",
            "Jira; Northstar Cloud Support Console; secrets manager",
            "Secret minimization; credential rotation; security escalation",
            "Security handled the suspected exposure and the support ticket was scrubbed of sensitive content.",
        ),
        ExpertCase(
            "A team wants to deploy a new API endpoint without a load test because the expected traffic is small.",
            "Build in small steps",
            "A small expected load still needs a measurable safe baseline. Start with a narrow internal test, observe limits, and expand gradually.",
            {"client_issue": 0, "bug": 0, "code": 1, "presentation": 0, "report": 0, "urgent": 0},
            "api_release_checklist.md",
            "1. Define latency, error-rate, and throughput targets.\n2. Test the endpoint with representative payloads.\n3. Confirm rate limits, timeouts, and logging.\n4. Release behind a controlled flag or cohort.\n5. Compare production metrics with the baseline before widening access.",
            "Postman; k6; Datadog; LaunchDarkly",
            "Baseline measurement; representative load testing; staged rollout",
            "The team found a timeout issue before launch and released the endpoint gradually after fixing it.",
        ),
        ExpertCase(
            "A new engineer proposes changing a customer-facing error message without checking translations or support macros.",
            "Follow standard process",
            "User-visible text is part of the product contract. Check localization, accessibility, documentation, and support references before publishing it.",
            {"client_issue": 1, "bug": 0, "code": 1, "presentation": 0, "report": 0, "urgent": 0},
            "release_communications_guide.md",
            "1. Find every UI and API occurrence of the message.\n2. Check translation keys and accessibility wording.\n3. Update support macros and documentation.\n4. Test the message in supported locales.\n5. Ask support and product to approve the final copy.",
            "GitHub; Lokalise; Storybook; Northstar Cloud Support Console",
            "Search-before-change; localization review; cross-team approval",
            "The message changed consistently across the product and support materials.",
        ),
        ExpertCase(
            "A customer asks whether a failed payment means their account will be disabled immediately.",
            "Follow standard process",
            "Billing communication must distinguish a failed attempt, retry window, account state, and actual suspension status using the ledger rather than assumptions.",
            {"client_issue": 1, "bug": 0, "code": 0, "presentation": 0, "report": 1, "urgent": 0},
            "billing_support_playbook.md",
            "1. Verify tenant and invoice IDs.\n2. Check payment status, retry schedule, and subscription state.\n3. State only what the ledger confirms.\n4. Explain the next billing action and deadline.\n5. Escalate disputes or incorrect suspension signals to billing operations.",
            "Stripe; Northstar Cloud billing ledger; Jira",
            "Ledger verification; state-machine reasoning; expectation setting",
            "The client received an accurate account-status explanation without an unsupported promise.",
        ),
        ExpertCase(
            "A support ticket has no reproduction steps, but the requester asks engineering to fix it urgently.",
            "Diagnose and verify",
            "Urgency does not replace the minimum evidence needed to act. Gather a small reproducible example while communicating that investigation has started.",
            {"client_issue": 1, "bug": 1, "code": 1, "presentation": 0, "report": 0, "urgent": 1},
            "engineering_handoff_template.md",
            "1. Ask for exact steps, timestamps, IDs, and expected behavior.\n2. Try to reproduce in a safe environment.\n3. Attach sanitized logs or a screen recording.\n4. State the current impact and urgency reason.\n5. Escalate with an explicit unknowns section if reproduction still fails.",
            "Northstar Cloud Support Console; Jira; Datadog",
            "Minimum reproducible example; evidence collection; unknowns tracking",
            "Engineering could act on a focused handoff instead of spending the first cycle reconstructing the report.",
        ),
        ExpertCase(
            "A client asks why a user can see a project but cannot edit it.",
            "Diagnose and verify",
            "Separate authentication from authorization. Confirm the user's role, project membership, inherited permissions, and whether the project is read-only by design.",
            {"client_issue": 1, "bug": 0, "code": 0, "presentation": 0, "report": 0, "urgent": 0},
            "permissions_support_runbook.md",
            "1. Capture tenant, user ID, project ID, and exact action that fails.\n2. Confirm the user is authenticated as the expected account.\n3. Compare role and project membership with a working user.\n4. Check inherited, team, and read-only permissions.\n5. Change access only through the approved admin workflow.",
            "Northstar Cloud Admin; Support Console; Jira",
            "Authentication-versus-authorization isolation; permission comparison; least privilege",
            "The team identified a missing project role and fixed access without granting unnecessary administrator permissions.",
        ),
        ExpertCase(
            "A client onboarding checklist is complete, but the client still cannot find the first required workflow.",
            "Follow standard process",
            "Completion of configuration does not prove successful adoption. Verify the user's first-run path and remove the specific discovery barrier.",
            {"client_issue": 1, "bug": 0, "code": 0, "presentation": 0, "report": 0, "urgent": 0},
            "client_onboarding_playbook.md",
            "1. Ask the client to share the exact screen and step where they stop.\n2. Verify the intended role and navigation permissions.\n3. Walk through one real workflow using their terminology.\n4. Capture the missing instruction or UI confusion.\n5. Update the onboarding checklist with the observed blocker.",
            "Northstar Cloud Console; Loom; Support Console",
            "First-run walkthrough; role-based validation; friction logging",
            "The client completed a real workflow and the onboarding guide gained a missing navigation step.",
        ),
        ExpertCase(
            "A nightly data sync finishes successfully but yesterday's records are missing from the customer dashboard.",
            "Diagnose and verify",
            "A successful job status only proves the process completed, not that the expected records arrived. Trace row counts and freshness through each stage.",
            {"client_issue": 1, "bug": 1, "code": 1, "presentation": 0, "report": 1, "urgent": 0},
            "data_sync_runbook.md",
            "1. Record the expected and actual record counts by tenant.\n2. Check source extraction time, transform logs, warehouse load, and dashboard refresh time.\n3. Compare one missing record across every stage.\n4. Check filters, late-arriving data, and duplicate suppression.\n5. Communicate whether the issue is delayed data, filtered data, or a failed stage.",
            "Airflow; Snowflake; Looker; Datadog",
            "Pipeline stage tracing; freshness checks; row-count reconciliation",
            "The team found a delayed dashboard refresh rather than a failed sync and corrected the refresh schedule.",
        ),
        ExpertCase(
            "A Python background job succeeds locally but fails in the staging environment because an environment variable is missing.",
            "Write and test",
            "Environment-dependent behavior should fail clearly and be covered by a startup check, not discovered halfway through a job.",
            {"client_issue": 0, "bug": 1, "code": 1, "presentation": 0, "report": 0, "urgent": 0},
            "engineering_runbook.md",
            "1. Capture the exact staging traceback and job version.\n2. Compare required configuration names without exposing secret values.\n3. Add a startup validation with a safe missing-variable message.\n4. Add a test for missing configuration and a smoke test with valid configuration.\n5. Update deployment documentation and environment templates.",
            "Python; pytest; GitHub Actions; Kubernetes",
            "Configuration validation; environment parity; smoke testing",
            "The job failed early with an actionable message and the deployment template documented the required variable.",
        ),
        ExpertCase(
            "A client reports intermittent 502 errors, but a single retry usually succeeds.",
            "Diagnose and verify",
            "A successful retry can hide a capacity, timeout, or upstream dependency problem. Measure the pattern rather than treating the retry as the fix.",
            {"client_issue": 1, "bug": 1, "code": 1, "presentation": 0, "report": 0, "urgent": 1},
            "incident_response_runbook.md",
            "1. Capture request IDs, endpoints, regions, timestamps, and retry counts.\n2. Compare 502 rate with latency, deployment, traffic, and upstream health.\n3. Check whether failures cluster by tenant or region.\n4. Use retry only as a temporary mitigation with a bounded limit.\n5. Escalate if the error rate, customer count, or data-safety risk grows.",
            "Datadog; Northstar Cloud API Gateway; PagerDuty",
            "Error-rate segmentation; upstream correlation; bounded retry policy",
            "The team found a regional upstream dependency pattern and avoided masking it with unlimited retries.",
        ),
        ExpertCase(
            "A product team wants to remove an old feature flag after the rollout appears stable.",
            "Build in small steps",
            "Flag cleanup is a code change and should confirm that no cohort, rollback, documentation, or support workflow still depends on the flag.",
            {"client_issue": 0, "bug": 0, "code": 1, "presentation": 0, "report": 0, "urgent": 0},
            "feature_flag_lifecycle.md",
            "1. Identify all code paths, cohorts, and default values using the flag.\n2. Check metrics for every rollout segment and recent support tickets.\n3. Confirm a rollback plan that does not depend on the flag.\n4. Remove the flag in a small pull request with tests.\n5. Delete stale configuration and update the release notes.",
            "LaunchDarkly; GitHub; pytest; Datadog",
            "Flag lifecycle review; dependency search; reversible cleanup",
            "The flag was removed without affecting a legacy cohort or losing the rollback path.",
        ),
        ExpertCase(
            "A customer asks for a data export in a format the product does not currently provide.",
            "Follow standard process",
            "Clarify the business purpose before promising a new format. An existing export or approved transformation may solve the need without a product change.",
            {"client_issue": 1, "bug": 0, "code": 0, "presentation": 0, "report": 1, "urgent": 0},
            "customer_response_templates.md",
            "1. Ask what system will receive the export and which fields are required.\n2. Check existing CSV, JSON, and scheduled export options.\n3. Offer a documented transformation if it does not change the data meaning.\n4. Record the request and business impact if a new format is needed.\n5. Do not promise a delivery date without product review.",
            "Northstar Cloud Analytics; Support Console; Jira",
            "Outcome clarification; supported-option mapping; scope control",
            "The client used an existing export with a documented transformation and no unsupported commitment was made.",
        ),
        ExpertCase(
            "A retrospective has a long list of complaints but no specific actions that can be checked later.",
            "Build in small steps",
            "A retrospective should convert observations into a few owned experiments or process changes with a measurable completion condition.",
            {"client_issue": 0, "bug": 1, "code": 0, "presentation": 0, "report": 1, "urgent": 0},
            "incident_review_template.md",
            "1. Group observations by contributing factor, not by who made the mistake.\n2. Select the two highest-leverage changes.\n3. Assign one owner and due date to each action.\n4. Define the metric or artifact that proves completion.\n5. Review the actions at the next team meeting.",
            "Notion; Jira; Datadog",
            "Blameless analysis; action prioritization; measurable follow-through",
            "The review produced two owned actions instead of an untracked list of complaints.",
        ),
        ExpertCase(
            "A client asks support to confirm that a new integration is secure before their security team reviews it.",
            "Escalate for review",
            "Support can describe documented controls but cannot certify a custom integration without security review and evidence.",
            {"client_issue": 1, "bug": 0, "code": 1, "presentation": 0, "report": 0, "urgent": 1},
            "security_pitch_guide.md",
            "1. Ask which controls and data flows the client needs evaluated.\n2. Use only current approved security documentation.\n3. Map the integration's data path without requesting unnecessary secrets.\n4. Route the questionnaire or exception to the security owner.\n5. Tell the client what is documented versus still under review.",
            "Trust Center; Northstar Cloud API; Jira",
            "Control mapping; data-flow review; evidence-based claims",
            "Security reviewed the integration and support avoided making an unsupported certification.",
        ),
        ExpertCase(
            "A team member wants to copy a successful client configuration into another tenant to save time.",
            "Diagnose and verify",
            "Configurations may contain tenant-specific identifiers, permissions, or assumptions. Compare the two environments before copying anything.",
            {"client_issue": 1, "bug": 0, "code": 1, "presentation": 0, "report": 0, "urgent": 0},
            "client_configuration_guide.md",
            "1. List the source configuration and tenant-specific values.\n2. Check whether the target has the same plan, roles, integrations, and data model.\n3. Remove secrets and identifiers from any template.\n4. Test the smallest configuration in a non-production scope.\n5. Get tenant-owner approval before applying the change broadly.",
            "Northstar Cloud Admin; Terraform; Jira",
            "Configuration diffing; secret removal; staged rollout",
            "The team reused the safe parts of the configuration without copying tenant-specific permissions.",
        ),
        ExpertCase(
            "A support engineer cannot find the answer in the runbooks and wants to mark the ticket as documentation complete.",
            "Follow standard process",
            "An unanswered question is evidence of a documentation gap, not proof that no procedure is needed. Capture the question and the eventual resolution.",
            {"client_issue": 1, "bug": 0, "code": 0, "presentation": 0, "report": 0, "urgent": 0},
            "support_triage_playbook.md",
            "1. Search the known runbooks, source files, and recent resolved cases.\n2. Record the exact unanswered question and search terms.\n3. Ask the owning team for the approved procedure.\n4. Resolve the client issue using reviewed evidence.\n5. Convert the answer into a dated runbook update with an owner.",
            "Northstar Cloud Support Console; Confluence; Jira",
            "Knowledge-gap capture; source verification; owned documentation update",
            "The ticket was resolved and the missing procedure became a maintained knowledge entry.",
        ),
        ExpertCase(
            "A customer requests a change that would affect only their tenant, but the requested behavior is not supported by the current permission model.",
            "Escalate for review",
            "A tenant-specific exception can create security and maintenance risk. Confirm the outcome, assess a supported workaround, and route exceptions through product and security review.",
            {"client_issue": 1, "bug": 0, "code": 1, "presentation": 0, "report": 0, "urgent": 0},
            "client_configuration_guide.md",
            "1. Clarify the desired business outcome.\n2. Confirm the limitation with the current permission model.\n3. Identify a supported workflow that meets most of the need.\n4. Document the impact and maintenance cost of an exception.\n5. Escalate for product and security approval before promising custom behavior.",
            "Northstar Cloud Admin; Jira; security review queue",
            "Outcome-based requirements; exception review; least-privilege design",
            "The client received a supported workaround while the exception request received the correct review.",
        ),
        ExpertCase(
            "A release candidate passes unit tests but fails an end-to-end test because the test data is stale.",
            "Write and test",
            "A passing unit suite does not prove the integrated workflow. Refresh deterministic test fixtures and make the end-to-end failure explainable.",
            {"client_issue": 0, "bug": 1, "code": 1, "presentation": 0, "report": 0, "urgent": 0},
            "engineering_runbook.md",
            "1. Capture the end-to-end failure and environment version.\n2. Check whether the fixture, seed data, or service contract changed.\n3. Rebuild a minimal deterministic fixture.\n4. Add an assertion that explains the expected workflow state.\n5. Run unit, integration, and end-to-end tests before release approval.",
            "Python; pytest; Playwright; GitHub Actions",
            "Layered testing; deterministic fixtures; contract verification",
            "The end-to-end test became repeatable and caught a real integration mismatch before release.",
        ),
        ExpertCase(
            "A team needs to communicate an incident update but the root cause is still unknown.",
            "Follow standard process",
            "An incident update should communicate impact, current mitigation, known facts, and next update time without guessing at root cause.",
            {"client_issue": 1, "bug": 1, "code": 0, "presentation": 0, "report": 0, "urgent": 1},
            "customer_response_templates.md",
            "1. State who is affected and what behavior is failing.\n2. Describe the current mitigation or investigation step.\n3. Separate confirmed facts from hypotheses.\n4. Give the next update time and communication owner.\n5. Update the message when the evidence changes.",
            "Statuspage; Northstar Cloud Support Console; PagerDuty",
            "Incident communication; fact-versus-hypothesis separation; time-bound updates",
            "Customers received useful updates without being told an unverified root cause.",
        ),
        ExpertCase(
            "A founder needs an investor PowerPoint that explains Northstar Cloud's value, traction, business model, and funding request in ten minutes.",
            "Build in small steps",
            "Investor presentations need a decision-oriented story: establish the customer problem, prove the solution and traction with verified evidence, explain the business model, and make a specific ask. Product detail belongs after the core investment case.",
            {"client_issue": 0, "bug": 0, "code": 0, "presentation": 1, "report": 1, "urgent": 0},
            "investor_presentation_playbook.md",
            "1. Define the investor audience, meeting length, and decision the deck should support.\n2. Build the narrative around problem, customer, solution, traction, market, business model, competition, team, and funding ask.\n3. Put one claim in each slide title and attach a source or calculation to every important number.\n4. Demonstrate the smallest workflow that proves the product value instead of listing every feature.\n5. End with the amount requested, planned use of funds, milestones, and next step.\n6. Move technical architecture and detailed metrics into an appendix.",
            "PowerPoint; Northstar Cloud Analytics; Snowflake; Northstar Cloud product demo environment",
            "Investor narrative; evidence-backed metrics; one-message-per-slide; progressive disclosure",
            "The team produced a focused ten-minute deck with a clear funding ask and an evidence appendix for follow-up questions.",
        ),
        ExpertCase(
            "A customer-success manager needs a quarterly business review deck for an enterprise client.",
            "Build in small steps",
            "A QBR should connect usage and support evidence to the customer's goals, then end with mutually owned actions rather than becoming a feature catalog.",
            {"client_issue": 1, "bug": 0, "code": 0, "presentation": 1, "report": 1, "urgent": 0},
            "qbr_presentation_playbook.md",
            "1. Confirm the customer's goals and review period.\n2. Show adoption, outcomes, wins, risks, and open issues with dates.\n3. Explain one insight per slide and compare against the agreed baseline.\n4. Connect product recommendations to a customer goal.\n5. End with three actions, owners, and dates.",
            "PowerPoint; Looker; Northstar Cloud Support Console",
            "Outcome-based storytelling; baseline comparison; joint action planning",
            "The QBR focused on customer outcomes and produced an agreed follow-up plan.",
        ),
        ExpertCase(
            "A product manager needs a roadmap presentation for leadership that includes too many competing initiatives.",
            "Build in small steps",
            "Leadership needs tradeoffs and sequencing, not a complete backlog. Group work by outcome, expose dependencies, and identify what will not be done.",
            {"client_issue": 0, "bug": 0, "code": 0, "presentation": 1, "report": 1, "urgent": 0},
            "roadmap_review_guide.md",
            "1. Group initiatives by customer or business outcome.\n2. Rank by impact, confidence, effort, and dependency.\n3. Show now, next, later, and explicitly deferred work.\n4. Put detailed tickets in an appendix.\n5. Ask leadership to approve the tradeoff, not just the slide design.",
            "PowerPoint; Jira; Productboard",
            "Outcome-based prioritization; dependency mapping; decision framing",
            "Leadership approved a smaller roadmap and documented the deferred work.",
        ),
        ExpertCase(
            "A sales engineer needs a product demo for a prospect that cares about reducing manual reporting work.",
            "Follow standard process",
            "The demo should mirror the prospect's workflow and prove one measurable improvement instead of showing every feature.",
            {"client_issue": 1, "bug": 0, "code": 0, "presentation": 1, "report": 1, "urgent": 0},
            "product_demo_playbook.md",
            "1. Confirm the prospect's current workflow and pain point.\n2. Prepare realistic but sanitized example data.\n3. Demonstrate the before-and-after path.\n4. Quantify the time or error reduction only when supported.\n5. Close with the next evaluation step and unanswered questions.",
            "PowerPoint; Northstar Cloud product demo environment; Looker",
            "Discovery-led demo; before-and-after proof; qualification questions",
            "The prospect saw a relevant workflow and the sales team captured a clear next step.",
        ),
        ExpertCase(
            "A customer asks how to rotate an API token without interrupting their integration.",
            "Follow standard process",
            "Credential rotation should use an overlap window: create and test the new token before revoking the old one, then verify usage has moved.",
            {"client_issue": 1, "bug": 0, "code": 1, "presentation": 0, "report": 0, "urgent": 1},
            "api_security_runbook.md",
            "1. Identify the tenant, integration, token owner, and maintenance constraints.\n2. Create the replacement token through the approved admin path.\n3. Update the integration securely and test a low-risk request.\n4. Monitor successful use of the new token.\n5. Revoke the old token and record the rotation evidence.",
            "Northstar Cloud Admin; secrets manager; Northstar Cloud API",
            "Credential overlap; least privilege; post-change verification",
            "The customer rotated credentials without an outage and the old token was revoked.",
        ),
        ExpertCase(
            "An API consumer receives HTTP 429 errors after increasing the number of parallel requests.",
            "Diagnose and verify",
            "First establish whether the limit is expected, tenant-specific, endpoint-specific, or caused by a retry storm before recommending a higher limit.",
            {"client_issue": 1, "bug": 1, "code": 1, "presentation": 0, "report": 0, "urgent": 0},
            "api_rate_limit_playbook.md",
            "1. Capture tenant, endpoint, timestamps, request rate, and response headers.\n2. Compare the limit with the customer's plan and documented policy.\n3. Check whether retries are synchronized or missing backoff.\n4. Recommend bounded exponential backoff and request batching.\n5. Escalate limit changes with measured demand and business impact.",
            "Northstar Cloud API Gateway; Datadog; Postman",
            "Rate-limit analysis; exponential backoff; load shaping",
            "The integration used backoff and batching instead of relying on an unsupported limit increase.",
        ),
        ExpertCase(
            "A new API endpoint returns all records on one request and times out for a large tenant.",
            "Write and test",
            "The endpoint needs bounded responses and a documented pagination contract before it can safely support large tenants.",
            {"client_issue": 1, "bug": 1, "code": 1, "presentation": 0, "report": 0, "urgent": 0},
            "api_contract_standards.md",
            "1. Reproduce with small, medium, and large datasets.\n2. Define page size, cursor behavior, ordering, and total-count semantics.\n3. Add tests for first page, next page, empty page, and invalid cursor.\n4. Add timeout and rate-limit behavior to the contract.\n5. Publish a migration example before enabling the endpoint broadly.",
            "Python; pytest; OpenAPI; Northstar Cloud API",
            "Pagination contract; boundary testing; large-tenant testing",
            "The endpoint handled large tenants predictably and the API documentation included pagination examples.",
        ),
        ExpertCase(
            "A client says a webhook integration stopped receiving events after their endpoint certificate expired.",
            "Diagnose and verify",
            "Confirm whether delivery failures began at certificate expiry and distinguish the endpoint's TLS problem from queue retry or event-subscription issues.",
            {"client_issue": 1, "bug": 1, "code": 1, "presentation": 0, "report": 0, "urgent": 0},
            "webhook_reliability_runbook.md",
            "1. Capture endpoint, subscription ID, certificate expiry time, and first failed delivery.\n2. Inspect delivery status and TLS error details.\n3. Confirm whether events are queued for retry or permanently failed.\n4. Have the client renew the certificate and test a signed event.\n5. Replay events only through the approved recovery process.",
            "Northstar Cloud Events API; Datadog; Postman",
            "Timeline correlation; TLS verification; controlled event replay",
            "The certificate was renewed and the team recovered events without duplicating downstream actions.",
        ),
        ExpertCase(
            "A client reports that a CSV import completes but several rows are missing.",
            "Diagnose and verify",
            "Import success means the job completed, not that every row was accepted. Reconcile input, validation rejects, deduplication, and final record counts.",
            {"client_issue": 1, "bug": 1, "code": 1, "presentation": 0, "report": 1, "urgent": 0},
            "data_import_runbook.md",
            "1. Preserve the original file and import job ID.\n2. Compare input rows, accepted rows, rejected rows, and duplicates.\n3. Inspect validation messages for the missing records.\n4. Test one rejected row after correcting its field format.\n5. Explain whether rows were rejected, deduplicated, or delayed.",
            "Northstar Cloud Importer; Snowflake; Support Console",
            "Row-count reconciliation; validation sampling; duplicate analysis",
            "The client received the rejected-row report and corrected the source file without losing accepted records.",
        ),
        ExpertCase(
            "A finance manager asks why monthly recurring revenue changed even though no new contracts were signed.",
            "Diagnose and verify",
            "Reconcile the metric definition and ledger events before attributing the change to growth or churn; upgrades, downgrades, renewals, credits, and currency can all affect it.",
            {"client_issue": 0, "bug": 1, "code": 0, "presentation": 0, "report": 1, "urgent": 0},
            "revenue_reporting_standards.md",
            "1. Define the MRR snapshot and comparison period.\n2. Reconcile contract, invoice, credit, renewal, and currency events.\n3. Group the change by expansion, contraction, churn, new business, and correction.\n4. Inspect a sample account for each group.\n5. Publish the bridge with definitions and data timestamp.",
            "Snowflake; Stripe; Looker",
            "Revenue bridge; event reconciliation; sample-account audit",
            "The team explained the change as a mix of renewals and credits instead of reporting an unsupported growth claim.",
        ),
        ExpertCase(
            "A support manager wants to measure response quality using only the number of tickets closed.",
            "Escalate for review",
            "Closure volume alone can reward rushed or reopened work. Agree on a balanced measure that includes resolution quality, customer outcome, and appropriate escalation.",
            {"client_issue": 1, "bug": 0, "code": 0, "presentation": 0, "report": 1, "urgent": 0},
            "support_metrics_guide.md",
            "1. Define the decision the metric should support.\n2. Compare closure volume with first response, resolution time, reopen rate, satisfaction, and escalation quality.\n3. Check for differences by issue severity and channel.\n4. Pilot a balanced scorecard with the support team.\n5. Ask the support owner to approve the interpretation before using it for performance decisions.",
            "Looker; Northstar Cloud Support Console; Snowflake",
            "Metric design; cohort normalization; incentive-risk review",
            "The team avoided a misleading single metric and adopted a balanced support scorecard.",
        ),
        ExpertCase(
            "A client requests a refund because they misunderstood a plan limit.",
            "Follow standard process",
            "Confirm the contract, usage, communication, and billing event before deciding whether this is a billing error, an exception, or a documentation issue.",
            {"client_issue": 1, "bug": 0, "code": 0, "presentation": 0, "report": 1, "urgent": 0},
            "billing_support_playbook.md",
            "1. Verify tenant, plan, invoice, and relevant usage.\n2. Compare the limit with the signed order and current documentation.\n3. Check whether the product message was clear and current.\n4. Route refund exceptions for approval instead of promising one.\n5. Record a documentation or UX follow-up if confusion was predictable.",
            "Stripe; Northstar Cloud billing ledger; Jira",
            "Contract-versus-usage review; exception approval; root-cause feedback",
            "Billing resolved the request consistently and product received a clear documentation improvement.",
        ),
        ExpertCase(
            "A renewal is at risk because the customer says they cannot prove the value of the product internally.",
            "Build in small steps",
            "Build a value narrative from the customer's goals and verified outcomes, not from a generic feature list.",
            {"client_issue": 1, "bug": 0, "code": 0, "presentation": 1, "report": 1, "urgent": 1},
            "renewal_value_playbook.md",
            "1. Confirm the original business goals and success criteria.\n2. Select a few verified outcomes with dates and baseline comparisons.\n3. Connect each outcome to the customer's stakeholder.\n4. Create a short value review with risks and next milestones.\n5. Ask the customer what evidence their renewal decision requires.",
            "Looker; PowerPoint; Northstar Cloud Support Console",
            "Outcome mapping; baseline comparison; stakeholder-specific storytelling",
            "The customer received an evidence-based value review and identified the remaining proof needed for renewal.",
        ),
        ExpertCase(
            "A manager needs concise meeting notes that turn a complicated discussion into accountable actions.",
            "Follow standard process",
            "Meeting notes should preserve decisions, unresolved questions, owners, and dates rather than transcribing every comment.",
            {"client_issue": 0, "bug": 0, "code": 0, "presentation": 0, "report": 1, "urgent": 0},
            "team_reporting_standards.md",
            "1. Record the meeting purpose and decision context.\n2. Separate decisions, actions, risks, and open questions.\n3. Assign one owner and due date to each action.\n4. Link supporting documents without duplicating them.\n5. Send notes for corrections and track the next review date.",
            "Notion; Jira; Google Calendar",
            "Decision logging; action ownership; concise documentation",
            "The notes became a usable follow-up artifact instead of a long transcript.",
        ),
        ExpertCase(
            "A team wants to add a new field to a customer report without checking whether downstream exports depend on the column order.",
            "Diagnose and verify",
            "A report change can break customer workflows even when the dashboard looks correct. Search consumers and test the output contract before changing it.",
            {"client_issue": 1, "bug": 1, "code": 1, "presentation": 0, "report": 1, "urgent": 0},
            "reporting_standards.md",
            "1. Find dashboards, exports, scheduled jobs, and customer scripts that consume the report.\n2. Check whether consumers rely on column names, order, or types.\n3. Add the field in a backward-compatible way when possible.\n4. Test representative exports and notify affected owners.\n5. Document the schema change and effective date.",
            "Looker; Snowflake; Northstar Cloud API; GitHub",
            "Consumer inventory; schema compatibility; representative export testing",
            "The field was added without silently breaking a customer export.",
        ),
        ExpertCase(
            "A new hire asks where to find the approved process for handling a high-value customer complaint.",
            "Follow standard process",
            "High-value complaints need consistent ownership and escalation, but the first response should still acknowledge impact and gather facts.",
            {"client_issue": 1, "bug": 0, "code": 0, "presentation": 0, "report": 0, "urgent": 1},
            "support_onboarding_curriculum.md",
            "1. Confirm the account owner and complaint impact.\n2. Acknowledge the concern without assigning blame.\n3. Capture timeline, previous contacts, and requested outcome.\n4. Notify the account owner and support lead through the escalation path.\n5. Give the customer a named owner and next update time.",
            "Northstar Cloud Support Console; Jira; account health dashboard",
            "Account-aware triage; structured empathy; ownership handoff",
            "The complaint had a clear owner and the customer received a time-bound next step.",
        ),
        ExpertCase(
            "A developer wants to add caching to an endpoint that returns customer-specific data.",
            "Escalate for review",
            "Caching can expose data across tenants or serve stale permissions. Define the cache key, invalidation rule, sensitivity, and review owner before enabling it.",
            {"client_issue": 1, "bug": 1, "code": 1, "presentation": 0, "report": 0, "urgent": 1},
            "api_security_runbook.md",
            "1. Identify whether the response contains tenant, user, or permission-specific data.\n2. Define a cache key that cannot cross tenant or user boundaries.\n3. Define TTL, invalidation, and authorization checks.\n4. Test permission changes and tenant isolation.\n5. Get security and platform review before production rollout.",
            "Redis; Python; pytest; Northstar Cloud API",
            "Cache-key design; tenant isolation testing; stale-data analysis",
            "The team rejected an unsafe global cache and designed a scoped cache with explicit invalidation.",
        ),
        ExpertCase(
            "A client wants to connect Northstar Cloud to a third-party CRM but has not identified which fields should sync.",
            "Build in small steps",
            "Start with the business workflow and a minimal field mapping before building a full bidirectional integration.",
            {"client_issue": 1, "bug": 0, "code": 1, "presentation": 0, "report": 0, "urgent": 0},
            "integration_planning_guide.md",
            "1. Identify the business event that should trigger the sync.\n2. Select the smallest set of required fields.\n3. Map ownership, format, update direction, and conflict behavior.\n4. Test one tenant with sanitized records.\n5. Add fields only after the first workflow is reliable.",
            "Northstar Cloud API; Salesforce sandbox; Postman",
            "Minimal integration; field mapping; conflict testing",
            "The team launched a narrow useful sync and avoided an ambiguous full-data integration.",
        ),
        ExpertCase(
            "A documentation page explains what a feature does but not when a support engineer should escalate it.",
            "Build in small steps",
            "Useful documentation needs decision boundaries, not just descriptions. Add symptoms, safe checks, stop conditions, owner, and evidence requirements.",
            {"client_issue": 1, "bug": 0, "code": 0, "presentation": 0, "report": 0, "urgent": 0},
            "support_triage_playbook.md",
            "1. Add the common symptom and supported scope.\n2. List low-risk checks a support engineer can perform.\n3. Define conditions that require escalation.\n4. Specify the handoff fields and source owner.\n5. Test the page with a new engineer using a realistic scenario.",
            "Confluence; Northstar Cloud Support Console; Jira",
            "Decision-tree documentation; stop conditions; new-hire usability testing",
            "The page became actionable and new engineers knew when to stop troubleshooting.",
        ),
        ExpertCase(
            "A customer-facing release note describes a performance improvement without showing how it was measured.",
            "Write and test",
            "Performance claims need a defined baseline, sample, environment, and metric; otherwise the note creates an expectation the team cannot defend.",
            {"client_issue": 1, "bug": 0, "code": 1, "presentation": 0, "report": 1, "urgent": 0},
            "release_communications_guide.md",
            "1. Identify the before-and-after version and workload.\n2. Define the metric and measurement window.\n3. Verify the result with engineering.\n4. State the scope and limitations in plain language.\n5. Remove or qualify the claim if evidence is incomplete.",
            "GitHub; Datadog; PowerPoint; customer docs portal",
            "Benchmark definition; evidence review; expectation management",
            "The release note made a defensible improvement claim with a clear scope.",
        ),
        ExpertCase(
            "A support engineer is unsure whether a customer request is a bug, a feature request, or a configuration issue.",
            "Diagnose and verify",
            "Classify the request by expected behavior, documentation, reproducibility, and supported configuration before assigning it to a team.",
            {"client_issue": 1, "bug": 1, "code": 1, "presentation": 0, "report": 0, "urgent": 0},
            "support_triage_playbook.md",
            "1. Ask what the customer expected and why.\n2. Check current documentation and plan capability.\n3. Reproduce with a supported configuration.\n4. Compare actual behavior with the approved product contract.\n5. Route as bug, configuration help, or feature request with the evidence that supports the classification.",
            "Northstar Cloud Support Console; Jira; product docs",
            "Expected-versus-actual classification; supported-scope check; evidence-based routing",
            "The request reached the correct owner with less back-and-forth between support and product.",
        ),
        ExpertCase(
            "SAML login fails for some users with a clock-skew error while others can sign in.",
            "Diagnose and verify",
            "Clock skew can affect assertion validity for only some devices or identity-provider paths. Compare timestamps and configuration before changing the SSO window.",
            {"client_issue": 1, "bug": 1, "code": 0, "presentation": 0, "report": 0, "urgent": 1},
            "identity_escalation_rules.md",
            "1. Capture tenant, affected user, IdP, device, and timestamp.\n2. Compare IdP assertion time, Northstar receipt time, and device clock.\n3. Check NTP and timezone configuration.\n4. Test with one affected and one working user.\n5. Escalate before widening clock tolerance because it changes security validation.",
            "Okta; SAML tracer; Northstar Cloud Admin",
            "Timestamp comparison; controlled user comparison; secure tolerance review",
            "The client corrected time synchronization instead of weakening the SAML validation window.",
        ),
        ExpertCase(
            "SCIM deprovisioning removed a user from the identity provider but the user still appears active in Northstar Cloud.",
            "Escalate for review",
            "Deprovisioning affects access and may require replaying a failed lifecycle event. Confirm the event, user mapping, and audit trail before manually changing status.",
            {"client_issue": 1, "bug": 1, "code": 1, "presentation": 0, "report": 0, "urgent": 1},
            "identity_lifecycle_runbook.md",
            "1. Capture tenant, external user ID, event ID, and deprovisioning time.\n2. Check SCIM delivery status and response code.\n3. Compare external ID mapping with the Northstar user record.\n4. Restrict access through the approved emergency path if risk is active.\n5. Escalate for event replay and audit confirmation.",
            "Okta; SCIM; Northstar Cloud Admin; Jira",
            "Lifecycle-event tracing; identity mapping; access containment",
            "The failed SCIM event was replayed and the audit record confirmed access removal.",
        ),
        ExpertCase(
            "Webhook deliveries fail after a customer rotates their signing secret, but the endpoint is reachable.",
            "Diagnose and verify",
            "Reachability does not prove signature compatibility. Compare the secret version, canonical payload, timestamp tolerance, and verification algorithm.",
            {"client_issue": 1, "bug": 1, "code": 1, "presentation": 0, "report": 0, "urgent": 0},
            "webhook_reliability_runbook.md",
            "1. Capture subscription ID, delivery ID, signature version, and error.\n2. Confirm which secret version the sender and receiver use.\n3. Recompute a signature on a sanitized payload using the documented canonical form.\n4. Check timestamp tolerance and clock synchronization.\n5. Rotate through an overlap window and replay one test event.",
            "Northstar Cloud Events API; Postman; customer logs",
            "HMAC verification; canonical-payload testing; secret overlap",
            "The integration accepted events after both sides used the same signature version and timestamp rules.",
        ),
        ExpertCase(
            "A customer custom domain shows a certificate warning after DNS was moved to a new provider.",
            "Diagnose and verify",
            "The DNS move may have changed the CNAME, validation record, or certificate renewal path. Verify DNS propagation and certificate coverage before reissuing anything.",
            {"client_issue": 1, "bug": 1, "code": 0, "presentation": 0, "report": 0, "urgent": 1},
            "custom_domain_runbook.md",
            "1. Capture hostname, tenant, DNS provider, and first warning time.\n2. Inspect CNAME, validation records, TTL, and current resolution from multiple regions.\n3. Compare the certificate SAN with the requested hostname.\n4. Check whether the certificate authority can complete validation.\n5. Escalate reissuance if DNS is correct but certificate state is stale.",
            "Cloudflare; dig; Northstar Cloud Admin; certificate manager",
            "DNS propagation; certificate-chain inspection; regional comparison",
            "The team corrected a missing validation record and restored the custom domain without changing application code.",
        ),
        ExpertCase(
            "A customer can load the app but a strict corporate firewall blocks the API hostname.",
            "Follow standard process",
            "Network allowlisting needs the supported hostname and port information, not a request to broadly allow all Northstar Cloud traffic.",
            {"client_issue": 1, "bug": 0, "code": 0, "presentation": 0, "report": 0, "urgent": 0},
            "network_connectivity_guide.md",
            "1. Capture the blocked hostname, port, region, and firewall message.\n2. Test DNS resolution and TLS separately.\n3. Provide the documented endpoint and required outbound port.\n4. Avoid recommending IP allowlists when addresses are dynamic.\n5. Confirm the customer can complete one safe API request after the change.",
            "dig; curl; Northstar Cloud API Gateway; customer firewall logs",
            "Layered connectivity checks; hostname allowlisting; least-broad network access",
            "The customer allowed the correct hostname and retained a narrow firewall policy.",
        ),
        ExpertCase(
            "A report shows a duplicated hour of usage on the night daylight-saving time ended.",
            "Diagnose and verify",
            "Local time repeats during the fallback transition. Reconcile using UTC and a timezone-aware event identifier before treating the extra hour as duplicate usage.",
            {"client_issue": 1, "bug": 1, "code": 0, "presentation": 0, "report": 1, "urgent": 0},
            "reporting_standards.md",
            "1. Record report timezone and daylight-saving transition date.\n2. Inspect UTC timestamps for events in the repeated local hour.\n3. Count unique event IDs rather than local-hour labels.\n4. Compare the dashboard and export transformation.\n5. Document whether the issue is display duplication or duplicate source events.",
            "Snowflake; Looker; Northstar Cloud Analytics",
            "Timezone-aware aggregation; unique-event reconciliation; boundary testing",
            "The report retained both valid hours and displayed the timezone definition clearly.",
        ),
        ExpertCase(
            "A CSV import turns accented customer names into unreadable characters.",
            "Diagnose and verify",
            "The file is likely encoded differently from the importer expectation. Inspect the byte-order mark and encoding before asking the customer to retype data.",
            {"client_issue": 1, "bug": 1, "code": 1, "presentation": 0, "report": 1, "urgent": 0},
            "data_import_runbook.md",
            "1. Preserve the original file and import ID.\n2. Inspect encoding, delimiter, quote, and byte-order mark.\n3. Test one accented row in a safe preview.\n4. Import as UTF-8 when supported and verify the stored value.\n5. Report any rows changed by the import before completing the job.",
            "Python; pandas; Northstar Cloud Importer; UTF-8 validator",
            "Encoding detection; fixture sampling; round-trip verification",
            "The importer handled UTF-8 correctly and the customer did not need to modify names.",
        ),
        ExpertCase(
            "An OAuth integration receives invalid_scope even though the client ID is correct.",
            "Diagnose and verify",
            "Client identity and permission scope are separate. Compare requested scopes with the app registration, tenant policy, and endpoint requirements.",
            {"client_issue": 1, "bug": 1, "code": 1, "presentation": 0, "report": 0, "urgent": 0},
            "oauth_integration_guide.md",
            "1. Capture tenant, client ID, requested scopes, and authorization endpoint.\n2. Compare requested scopes with the registered app and approved consent.\n3. Remove unnecessary scopes and test the minimum required set.\n4. Check whether an admin-consent policy blocks the scope.\n5. Document the final token claims and endpoint access.",
            "OAuth; Northstar Cloud Admin; Postman",
            "Minimum-scope testing; consent-policy review; token-claim inspection",
            "The integration requested only approved scopes and passed tenant-admin consent.",
        ),
        ExpertCase(
            "A customer asks why a scheduled report ran one hour later after a regional deployment.",
            "Diagnose and verify",
            "Scheduled jobs can be affected by region, timezone, daylight-saving rules, queue delay, or deployment migration. Trace the schedule rather than assuming the deployment broke it.",
            {"client_issue": 1, "bug": 1, "code": 1, "presentation": 0, "report": 1, "urgent": 0},
            "scheduled_reporting_runbook.md",
            "1. Capture tenant schedule, timezone, region, job ID, and expected time.\n2. Compare scheduler trigger, queue time, worker start, and delivery time.\n3. Check deployment configuration and timezone defaults.\n4. Reproduce with a non-customer test schedule.\n5. Communicate whether the change affects trigger time or only delivery time.",
            "Airflow; Datadog; Snowflake; Northstar Cloud Analytics",
            "Schedule-stage tracing; regional comparison; timezone verification",
            "The team identified a region default and corrected the schedule without changing report data.",
        ),
        ExpertCase(
            "A backup restore completes but the restored tenant is missing a recently uploaded attachment.",
            "Escalate for review",
            "Restore completion does not guarantee point-in-time coverage for every storage system. Confirm backup timestamp, object-store version, and data-loss scope before telling the customer the restore is complete.",
            {"client_issue": 1, "bug": 1, "code": 0, "presentation": 0, "report": 0, "urgent": 1},
            "backup_restore_runbook.md",
            "1. Identify tenant, attachment ID, upload time, and restore point.\n2. Check database record and object-store version separately.\n3. Compare backup coverage with the requested recovery point.\n4. Do not overwrite the restored tenant while scope is unknown.\n5. Escalate to the recovery owner with the exact missing object and impact.",
            "S3; PostgreSQL; restore console; Jira",
            "Point-in-time recovery; cross-system reconciliation; data-loss containment",
            "The team measured the recovery gap accurately and avoided claiming a complete restore prematurely.",
        ),
        ExpertCase(
            "A screen-reader user cannot identify which field caused a form submission error.",
            "Write and test",
            "Accessible validation must associate the error with the field and announce it without relying on color or visual position.",
            {"client_issue": 1, "bug": 1, "code": 1, "presentation": 0, "report": 0, "urgent": 0},
            "accessibility_review_guide.md",
            "1. Reproduce with keyboard navigation and a screen reader.\n2. Check label, focus, error association, and live-region behavior.\n3. Add an accessible test for the failing validation state.\n4. Verify color contrast and non-visual instructions.\n5. Re-test the full form after the smallest fix.",
            "React; Storybook; axe; Playwright",
            "WCAG validation; keyboard testing; accessible error association",
            "The form announced the specific error and passed the accessibility regression test.",
        ),
        ExpertCase(
            "A client says an export opens with formulas instead of values in their spreadsheet tool.",
            "Follow standard process",
            "Clarify whether the client needs a raw data export, a calculated report, or a presentation-ready file; do not silently change the export semantics.",
            {"client_issue": 1, "bug": 0, "code": 0, "presentation": 0, "report": 1, "urgent": 0},
            "customer_response_templates.md",
            "1. Ask which spreadsheet tool and output purpose they use.\n2. Inspect whether the export intentionally contains formulas or calculated values.\n3. Offer the documented export type that matches the purpose.\n4. Test a small file in the customer's environment if possible.\n5. Record a product request only if a supported format cannot meet the need.",
            "Northstar Cloud Analytics; CSV exporter; Excel-compatible export",
            "Outcome clarification; format comparison; small-file validation",
            "The customer selected the correct export type without changing the underlying report calculation.",
        ),
        ExpertCase(
            "An incident alert fires because a health check is failing, but customer traffic appears normal.",
            "Diagnose and verify",
            "A health check may fail for a dependency, probe location, certificate, or overly strict threshold. Compare probe failure with real customer signals before declaring an outage.",
            {"client_issue": 0, "bug": 1, "code": 1, "presentation": 0, "report": 0, "urgent": 1},
            "incident_response_runbook.md",
            "1. Record probe region, endpoint, status, and first failure.\n2. Compare synthetic probe results with request success, latency, and error rate.\n3. Check dependency, certificate, and threshold changes.\n4. Avoid suppressing the alert before the failure mode is understood.\n5. Fix the probe or escalate a real partial outage with evidence.",
            "Datadog; PagerDuty; Northstar Cloud API Gateway",
            "Synthetic-versus-real traffic comparison; probe diagnosis; alert-quality review",
            "The team corrected a regional probe configuration while preserving meaningful customer-impact alerts.",
        ),
        ExpertCase(
            "A customer requests a deletion, but the records are referenced by an unresolved billing dispute.",
            "Escalate for review",
            "Deletion, retention, audit, and dispute obligations can conflict. Pause the irreversible action and obtain data-owner and legal/privacy guidance.",
            {"client_issue": 1, "bug": 0, "code": 0, "presentation": 0, "report": 1, "urgent": 1},
            "privacy_request_runbook.md",
            "1. Capture tenant, record scope, request date, and dispute reference.\n2. Check retention, legal hold, billing, and audit requirements.\n3. Do not delete or export the records manually.\n4. Route the conflict to privacy and billing owners.\n5. Communicate that scope is under review and provide the next update date.",
            "Jira; Snowflake; privacy request queue; billing ledger",
            "Retention conflict review; scope minimization; irreversible-action control",
            "The request was handled consistently without destroying records needed for the dispute.",
        ),
        ExpertCase(
            "A release uses a new third-party dependency with a license that the team has not reviewed.",
            "Escalate for review",
            "Dependency functionality is not enough for release approval. Review license, security, maintenance, and transitive dependency risk before shipping.",
            {"client_issue": 0, "bug": 0, "code": 1, "presentation": 0, "report": 0, "urgent": 0},
            "engineering_review_standards.md",
            "1. Record package, version, source, and intended use.\n2. Review direct and transitive licenses.\n3. Check security advisories and maintenance activity.\n4. Ask legal/security to approve exceptions.\n5. Pin the approved version and document upgrade ownership.",
            "GitHub; Dependabot; SBOM scanner",
            "License review; software supply-chain assessment; version pinning",
            "The team replaced an unclear dependency with an approved alternative and recorded ownership.",
        ),
    ]
    for case in cases:
        inferred_software, inferred_methods = infer_company_metadata(case.summary)
        case.software = case.software or inferred_software
        case.methods = case.methods or inferred_methods
    # Add a broad, deterministic synthetic knowledge pack for the demo. These
    # cases are intentionally generated from realistic Northstar Cloud task
    # variations so the prototype has enough coverage for many different ways
    # a new employee might describe the same kind of problem.
    return cases + build_generated_cases()


def build_generated_cases() -> list[ExpertCase]:
    """Create 540 varied expert decisions for the fictional demo company.

    The generator keeps the data reproducible while varying the wording,
    affected scope, trigger, and verification detail. This gives the local
    retrieval model more examples without requiring a large checked-in file.
    """
    variants = [
        ("single workspace", "one Northstar Cloud workspace", "the workspace owner"),
        ("one customer", "one customer tenant", "the account owner"),
        ("a pilot group", "a pilot group of users", "the pilot lead"),
        ("a regional rollout", "customers in one region", "the regional owner"),
        ("a new release", "the newest production release", "the release owner"),
        ("a legacy workflow", "an older supported workflow", "the support lead"),
        ("a high-value account", "a strategic customer account", "the account team"),
        ("an internal team", "an internal Northstar Cloud team", "the team manager"),
        ("a scheduled job", "one scheduled automation", "the data owner"),
        ("a customer-facing report", "one customer-facing report", "the reporting owner"),
        ("an API integration", "one external API integration", "the integration owner"),
        ("a security-sensitive tenant", "one security-sensitive tenant", "the security owner"),
        ("a renewal cycle", "one customer renewal cycle", "the customer success lead"),
        ("a new hire", "one new employee workflow", "the onboarding owner"),
        ("a time-sensitive request", "a request due today", "the incident lead"),
        ("a weekend change window", "a weekend maintenance window", "the on-call engineer"),
        ("a regulated customer", "a regulated customer tenant", "the compliance owner"),
        ("a small business account", "a small business tenant", "the support lead"),
        ("an enterprise account", "an enterprise tenant with custom controls", "the enterprise lead"),
        ("a mobile user", "a mobile client workflow", "the client platform owner"),
        ("a browser workflow", "a browser-based user workflow", "the frontend owner"),
        ("a webhook consumer", "one webhook consumer", "the integration owner"),
        ("a batch export", "one scheduled export", "the analytics owner"),
        ("a quarterly review", "a quarterly customer review", "the customer success lead"),
        ("a renewal presentation", "a renewal presentation for one account", "the account executive"),
        ("a board update", "an internal board update", "the executive sponsor"),
        ("a launch checklist", "one product launch checklist", "the launch manager"),
        ("a rollback decision", "one possible production rollback", "the incident commander"),
        ("a certificate rotation", "one certificate rotation", "the platform owner"),
        ("a permission change", "one requested permission change", "the security approver"),
        ("a data correction", "one requested data correction", "the data steward"),
        ("a duplicate event", "one duplicated billing or webhook event", "the systems owner"),
        ("a delayed notification", "one delayed customer notification", "the messaging owner"),
        ("a failed backup", "one failed backup job", "the reliability owner"),
        ("a restore test", "one scheduled restore test", "the infrastructure owner"),
        ("an accessibility review", "one accessibility review", "the design-system owner"),
        ("a localization update", "one localized customer workflow", "the content owner"),
        ("a documentation migration", "one documentation migration", "the documentation owner"),
        ("a training session", "one employee training session", "the enablement lead"),
        ("a support queue spike", "a sudden support queue spike", "the support manager"),
        ("a recurring incident", "a recurring production incident", "the reliability lead"),
        ("a new vendor", "one new third-party vendor integration", "the procurement owner"),
        ("a dependency update", "one dependency upgrade", "the engineering owner"),
        ("a schema change", "one planned data schema change", "the data platform owner"),
        ("an audit request", "one audit evidence request", "the compliance lead"),
        ("a privacy request", "one customer privacy request", "the privacy owner"),
        ("a contract exception", "one requested contract exception", "the legal owner"),
        ("a high-severity ticket", "a high-severity customer ticket", "the escalation manager"),
        ("a post-incident review", "one post-incident review", "the incident commander"),
        ("a forecast update", "one customer or revenue forecast", "the finance owner"),
    ]
    domains = [
        {
            "topic": "API request failures",
            "keywords": "API endpoint request timeout 429 response payload",
            "label": "Diagnose and verify",
            "reason": "The former expert reproduced the request, separated client, gateway, and service behavior, and used the exact response and timestamp before changing anything.",
            "steps": "1. Capture endpoint, method, tenant, request ID, timestamp, and sanitized payload.\n2. Reproduce with the smallest request that still fails.\n3. Compare client, gateway, and service logs.\n4. Check rate limits, timeout settings, and the last known-good release.\n5. Document the confirmed cause before proposing a narrow fix.",
            "software": "Northstar Cloud API Gateway; Postman; Datadog",
            "methods": "Reproduction-first debugging; request tracing; boundary testing",
            "source": "api_troubleshooting_playbook.md",
            "outcome": "The team isolated the failing layer and avoided changing unrelated services.",
        },
        {
            "topic": "Python and service code changes",
            "keywords": "Python function code exception unit test regression",
            "label": "Write and test",
            "reason": "The former expert defined expected behavior, made the smallest code change, and protected it with a focused test before expanding the change.",
            "steps": "1. Write the expected input, output, and edge cases.\n2. Reproduce the current behavior with a small test.\n3. Make the smallest readable change.\n4. Run unit, error-path, and regression tests.\n5. Record the changed files and remaining risks in the handoff.",
            "software": "Python; pytest; GitHub Actions",
            "methods": "Test-driven change; small diffs; regression testing",
            "source": "engineering_code_standards.md",
            "outcome": "The change passed focused tests and was easier for another engineer to review.",
        },
        {
            "topic": "Client support and communication",
            "keywords": "client customer user tenant support impact update response",
            "label": "Follow standard process",
            "reason": "The former expert confirmed impact and ownership first, then gave the customer a specific update time instead of promising an unverified fix.",
            "steps": "1. Confirm tenant, affected users, impact, timeline, and attempted steps.\n2. Reproduce or verify the issue with a low-risk check.\n3. Separate confirmed facts from hypotheses.\n4. Send the client the next action, owner, and update time.\n5. Escalate if the issue repeats or affects multiple customers.",
            "software": "Northstar Cloud Support Console; Jira; customer timeline",
            "methods": "Impact assessment; structured client update; evidence-based handoff",
            "source": "support_triage_playbook.md",
            "outcome": "The client received a clear update while the internal team continued investigation.",
        },
        {
            "topic": "Reporting and analytics",
            "keywords": "report dashboard metric KPI CSV retention data discrepancy",
            "label": "Diagnose and verify",
            "reason": "The former expert reconciled definitions, filters, timezones, and one sample before publishing a conclusion about the metric.",
            "steps": "1. Record metric definition, date range, timezone, filters, and scope.\n2. Compare the dashboard with the source export on one small sample.\n3. Check joins, duplicate rows, late-arriving data, and refresh time.\n4. Label confirmed numbers separately from hypotheses.\n5. Ask the data owner to review unexplained discrepancies.",
            "software": "Northstar Cloud Analytics; Snowflake; Looker",
            "methods": "Metric reconciliation; timezone checks; sample validation",
            "source": "reporting_standards.md",
            "outcome": "The team corrected the interpretation or documented the data limitation before sharing the report.",
        },
        {
            "topic": "Investor and customer presentations",
            "keywords": "presentation slide deck investor pitch PowerPoint story metrics",
            "label": "Follow standard process",
            "reason": "The former expert built the story around the audience's decision, used verified evidence, and removed detail that did not support the main message.",
            "steps": "1. Define the audience, decision, and one message.\n2. Organize the story around the audience's problem and the product outcome.\n3. Give every slide one conclusion and one supporting visual.\n4. Verify metrics, dates, and assumptions with the source owner.\n5. End with the exact ask, owner, milestone, and next date.",
            "software": "PowerPoint; Northstar Cloud demo environment; Looker",
            "methods": "Audience-first storytelling; one-message-per-slide; evidence review",
            "source": "presentation_standards.md",
            "outcome": "The presentation was easier to follow and the audience understood the requested decision.",
        },
        {
            "topic": "Security and access requests",
            "keywords": "security access permission SSO SAML secret production privacy",
            "label": "Escalate for review",
            "reason": "The former expert paused irreversible or privilege-expanding actions and routed the request to the designated security or data owner.",
            "steps": "1. Identify tenant, system, scope, requester, and business need.\n2. Do not copy secrets or sensitive records into a broad ticket.\n3. Check least-privilege, approval, retention, and audit requirements.\n4. Preserve only the evidence the responsible owner needs.\n5. Obtain security or privacy approval before making the final change.",
            "software": "Northstar Cloud Admin; Okta; Jira; audit log",
            "methods": "Least-privilege review; scope minimization; approval workflow",
            "source": "security_access_runbook.md",
            "outcome": "The request was resolved with the right approval trail and without exposing unnecessary data.",
        },
        {
            "topic": "Deployment and reliability",
            "keywords": "deployment release latency outage rollback monitoring production",
            "label": "Diagnose and verify",
            "reason": "The former expert compared the new release with the last known-good version, measured customer impact, and kept rollback reversible.",
            "steps": "1. Record release version, start time, affected region, and customer impact.\n2. Compare error rate, latency, and saturation with the last known-good window.\n3. Check logs, dependency health, feature flags, and recent configuration changes.\n4. Use a canary or rollback only with an owner and success condition.\n5. Write the incident timeline and follow-up action.",
            "software": "GitHub Actions; Datadog; PagerDuty; Northstar Cloud API",
            "methods": "Canary comparison; error-budget review; reversible rollback",
            "source": "incident_response_runbook.md",
            "outcome": "The team reduced customer impact while preserving evidence for the root-cause review.",
        },
        {
            "topic": "Billing and operations",
            "keywords": "billing invoice payment subscription renewal revenue operations",
            "label": "Follow standard process",
            "reason": "The former expert reconciled the invoice, subscription state, payment event, and account owner before changing a billing record.",
            "steps": "1. Capture tenant, invoice, subscription, payment event, and requested correction.\n2. Compare the billing ledger with the product entitlement state.\n3. Check refunds, credits, renewal dates, and duplicate events.\n4. Do not edit the ledger without the billing owner's approval.\n5. Communicate the confirmed status and next update time.",
            "software": "Northstar Cloud Billing; Stripe test console; Snowflake; Jira",
            "methods": "Ledger reconciliation; event timeline review; approval control",
            "source": "billing_operations_playbook.md",
            "outcome": "The billing status was corrected or explained without creating a second accounting discrepancy.",
        },
        {
            "topic": "Onboarding and documentation",
            "keywords": "onboarding documentation runbook new hire workflow training checklist",
            "label": "Build in small steps",
            "reason": "The former expert started with the smallest complete workflow, tested it with a new hire, and improved the documentation from observed confusion.",
            "steps": "1. Define the new employee's first successful outcome.\n2. List prerequisites, access, examples, and the owner for each step.\n3. Build a short path that works end to end.\n4. Ask a new person to follow it without coaching and record blockers.\n5. Update the runbook and add a dated owner for future review.",
            "software": "Notion; Northstar Cloud Admin; GitHub; Loom",
            "methods": "Smallest useful workflow; newcomer test; documentation review",
            "source": "onboarding_documentation_guide.md",
            "outcome": "A new employee completed the workflow with fewer live interruptions and clearer escalation points.",
        },
        {
            "topic": "Data pipeline and scheduled jobs",
            "keywords": "data pipeline ETL scheduled job warehouse refresh missing rows",
            "label": "Diagnose and verify",
            "reason": "The former expert traced the job from input to warehouse, compared row counts and timestamps, and identified whether the issue was late data or a failed transformation.",
            "steps": "1. Capture job run ID, source window, expected rows, and actual rows.\n2. Check scheduler, source freshness, transformation logs, and warehouse load status.\n3. Compare one affected partition with the prior successful run.\n4. Re-run only the safe, idempotent step if approved.\n5. Document data impact and notify report owners before refresh.",
            "software": "Snowflake; dbt; GitHub Actions; Datadog",
            "methods": "Lineage tracing; row-count reconciliation; idempotent rerun",
            "source": "data_pipeline_runbook.md",
            "outcome": "The missing data source was identified and the report owner received an accurate refresh estimate.",
        },
        {
            "topic": "Product and process improvement",
            "keywords": "product request workflow process improvement feature prioritization feedback",
            "label": "Build in small steps",
            "reason": "The former expert converted the request into a measurable small experiment before committing to a broad product change.",
            "steps": "1. Define the user problem and measurable success condition.\n2. Separate the desired outcome from the requested feature.\n3. Test the smallest reversible workflow with a representative user.\n4. Measure adoption, failure points, and support cost.\n5. Decide whether evidence supports a larger build or a different solution.",
            "software": "Jira; Northstar Cloud product analytics; Figma",
            "methods": "Problem framing; smallest experiment; outcome measurement",
            "source": "product_discovery_playbook.md",
            "outcome": "The team learned whether the underlying problem was real before spending effort on a large feature.",
        },
        {
            "topic": "Client response and handoff",
            "keywords": "client email response escalation handoff customer communication next update",
            "label": "Follow standard process",
            "reason": "The former expert wrote a response that distinguished confirmed facts, unknowns, next action, owner, and update time.",
            "steps": "1. Summarize the customer's request and business impact in one sentence.\n2. State what has been verified and what is still unknown.\n3. Give one safe next action and the responsible owner.\n4. Set the next update time without promising an unverified fix.\n5. Attach a concise handoff with logs, timestamps, scope, and reproduction.",
            "software": "Northstar Cloud Support Console; Jira; approved response templates",
            "methods": "Fact-versus-hypothesis separation; concise handoff; expectation setting",
            "source": "customer_response_templates.md",
            "outcome": "The customer received a useful update and engineering could start without repeating discovery work.",
        },
    ]

    generated: list[ExpertCase] = []
    for domain_index, domain in enumerate(domains):
        for variant_index, (variant_name, scope, owner) in enumerate(variants):
            summary = f"Northstar Cloud {domain['topic']} task for {scope}: the team needs help with {domain['keywords']}."
            reasoning = f"{domain['reason']} The current variation affects {scope}, and {owner} needs a clear, traceable decision."
            features = infer_features(f"{summary} {domain['keywords']}")
            features.update({"age_days": 0, "defect": float("bug" in domain["keywords"] or "failure" in domain["keywords"]), "evidence": 1.0, "repeat_customer": float("customer" in scope), "late_request": float("today" in scope), "high_value": float("high-value" in scope)})
            generated.append(ExpertCase(
                summary,
                domain["label"],
                reasoning,
                features,
                f"generated_{domain_index + 1:02d}_{variant_index + 1:02d}_{domain['source']}",
                domain["steps"],
                domain["software"],
                f"{domain['methods']}; variation: {variant_name}",
                domain["outcome"],
            ))
    return generated


def infer_company_metadata(text: str) -> tuple[str, str]:
    """Attach fictional Northstar Cloud tools and methods to demo cases."""
    lowered = text.lower()
    if "upload" in lowered:
        return "Northstar Cloud Console; Northstar Cloud API Gateway; Datadog", "Boundary testing; log correlation; request tracing"
    if "dashboard" in lowered or "retention" in lowered or "report" in lowered or "metric" in lowered:
        return "Northstar Cloud Analytics; Snowflake; Looker", "Metric reconciliation; timezone checks; cohort analysis"
    if "sso" in lowered or "login" in lowered or "identity" in lowered:
        return "Northstar Cloud Admin; Okta; SAML", "Identity-provider trace review; tenant isolation; least-privilege checks"
    if "deployment" in lowered or "latency" in lowered:
        return "GitHub Actions; Datadog; Northstar Cloud API", "Canary comparison; error-budget review; rollback procedure"
    if "python" in lowered or "api" in lowered or "function" in lowered or "code" in lowered:
        return "Python; pytest; Northstar Cloud API", "Reproduction-first debugging; contract testing; regression testing"
    if "presentation" in lowered or "slide" in lowered or "pitch" in lowered:
        return "PowerPoint; Northstar Cloud product demo environment", "Audience-first storytelling; one-message-per-slide; progressive disclosure"
    if "client" in lowered or "customer" in lowered or "support" in lowered:
        return "Northstar Cloud Support Console; Jira", "Impact assessment; structured client updates; escalation handoff"
    return "Northstar Cloud internal tools", "Clarify, reproduce, verify, document"
