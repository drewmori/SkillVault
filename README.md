# SkillVault

SkillVault saves the decision making of senior employees so new engineers and workers can solve customer problems faster, with traceable evidence and safe general suggestions when the company's documentation is incomplete or the employee doesn't know what to do next.

The competition demo uses a fictional SaaS company called **Northstar Cloud**. Its former expert knowledge covers client troubleshooting, engineering handoffs, code help, escalation decisions, client outreach, reporting, presentations, security, billing, onboarding, and company procedures.

## Competition documentation

### How Codex and GPT-5.6 were used

Codex was used as the primary development environment and coding collaborator. It helped build the Streamlit interface, organize the data model, implement the TF-IDF retrieval and scikit-learn classification pipeline, debug the small-dataset training issue, add expert-decision ingestion, expand the fictional knowledge pack, add media support, improve the tailoring logic, and refine the user experience.

GPT-5.6 was used during development to plan the product concept, compare competition categories, design the expert-knowledge workflow, generate and review realistic Northstar Cloud scenarios, identify missing edge cases, debug implementation decisions, and improve the product language and visual flow. The final app does not make an OpenAI API call and does not require an API key; its runtime is intentionally local and transparent.

### Work completed during the submission period

The submission-period work beginning July 13, 2026 included:

- Turning the initial concept into the Northstar Cloud support and engineering use case.
- Expanding the demo knowledge base to 97 expert decision cases across technical support, code, APIs, incidents, reporting, presentations, security, privacy, billing, client communication, onboarding, permissions, data pipelines, sales, documentation, accessibility, identity, networking, and escalation.
- Adding detailed expert instructions, reasoning, outcomes, software, methods, source files, and media attachments.
- Adding CSV import and an approval step for new expert decisions.
- Adding the Add Expert Decision workflow for actions recorded after the work happened.
- Adding Assist Mode source tracing, confidence signals, escalation guidance, and general suggestions.
- Adding tailored directions that preserve historical expert judgment while adapting the steps to the new issue.
- Adding Learn Mode, feedback-based correction, model insights, and a more complete visual product experience.

### How judges can test it

The app uses fictional Northstar Cloud data, so no account, API key, private company file, or external service is required.

1. Install Python 3.11 or newer.
2. Open a terminal in the repository folder.
3. Install dependencies with the command below.
4. Start Streamlit with the command below.
5. Open the local URL shown in the terminal.
6. In **Assist Mode**, analyze the prefilled upload-limit case and compare the historical evidence with the tailored SkillVault answer.
7. In **Add Expert Decision**, add an approved decision with tools, methods, outcome, and optional media, then test a related issue in Assist Mode.
8. In **Import Expert Data**, upload `sample_data/northstar_cloud_cases.csv`, review the editable preview, approve it, and retrain the model.
9. In **Learn Mode**, answer a practice case and compare the response with the expert's process.

The project is intended to run as a local Streamlit application. Approved decisions from **Add Expert Decision**, approved CSV imports, human corrections, and approved final plans are saved locally in JSON files next to the app so they remain available after restart. Those local files are ignored by Git because they may contain company-specific information; the fictional sample knowledge pack remains stored in the repository.

## Current prototype

- **Assist Mode:** enter a client issue or company task and receive a detailed playbook, source files, similar expert cases, escalation guidance, and general suggestions when the evidence is weak.
- **SkillVault answer writer:** automatically turns retrieved professional decisions into a detailed playbook, client-response draft, engineering-handoff draft, escalation guidance, and clearly labeled general suggestions.
- **Learn Mode:** solve a realistic Northstar Cloud task in your own words and compare your process with the former expert's instructions.
- **Add Expert Decision:** record an expert action after it happened, attach screenshots, diagrams, or screen recordings, confirm it is accurate, and retrain SkillVault immediately.
- **Import Expert Data:** upload support tickets, runbooks, reports, client templates, or case histories as CSV, review the rows, and approve them before retraining.
- **Model Insights:** inspect the cases, decision classes, and validation score.

## Run in VS Code

From the project folder:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

If dependencies have not been installed yet:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

GPT-5.6 and Codex were used during development. The submitted demo runs locally without an API key: scikit-learn retrieves and classifies the expert cases, and SkillVault's answer writer formats the evidence into useful next steps.

## Sample knowledge pack

The `sample_data/` folder contains the fictional Northstar Cloud source files and `northstar_cloud_cases.csv`. Judges can upload the CSV through **Import Expert Data** to test the knowledge-ingestion flow.

## Upload format

CSV uploads require `case`, `decision`, and `reasoning`. Optional columns are `instructions`, `source_file`, `software`, `methods`, and `outcome`. The Add Expert Decision form collects the same fields for one action at a time. `software` and `methods` let the model learn the exact tools and procedures associated with each expert example.

Valid decision approaches are:

- `Follow standard process`
- `Diagnose and verify`
- `Build in small steps`
- `Write and test`
- `Escalate for review`

The app displays an editable preview and does not retrain until the user approves the imported rows.

## Safety and evidence behavior

SkillVault separates company-specific knowledge from general suggestions. When historical evidence is weak or no close case exists, it warns the user, labels general suggestions clearly, and recommends human review for incidents, security, data integrity, or ambiguous cases.

Attached media is linked to the expert case. When a related case is retrieved in Assist Mode, its images and videos appear with the written instructions and source file.
