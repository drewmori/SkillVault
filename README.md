# SkillVault



## Current prototype

**Feedback that affects future answers:** The feedback form now includes **Too vague**, **Not specific enough**, **Too short**, **Too long**, and **Not helpful**. With **Use this feedback to improve future answers in this company** enabled, new Chat and Assist Mode answers apply saved, bounded answer-writing preferences. Low clarity/actionability ratings also affect those preferences. Negative feedback on questions with overlapping keywords requests an applicability recheck, not automatic source removal. Free-text notes and suggested fixes are not inserted into future prompts as policy; factual corrections still require expert approval. The latest submission per reviewer/question replaces repeated votes; only the latest 100 opted-in events are considered. Later length preferences override earlier ones. Administrators/expert reviewers can inspect and reset the learned preferences in Model Insights without deleting the audit log. Older feedback without the opt-in field is not retroactively applied. This is persistent feedback-conditioned generation, not neural-network retraining. The rules-based offline fallback adds a limited execution checklist or compact formatting; it does not gain new reasoning abilities. Automated tests verify isolation, persistence, reset, prompt changes, and preservation of the company-data-only section; actual model improvement still needs human evaluation.

**Two separate recommendations:** New chat and Assist Mode answers first show a company-data-only section quoting recorded approaches, steps, reasoning, constraints, and exceptions from up to three retrieved decisions. No generative rewriting is used in that section. Weak matches are explicitly marked, and an empty search does not produce an invented company recommendation. The second section combines relevant company evidence with adapted/general guidance, with the local model instructed to identify additions separately from source-backed actions. When the model is unavailable, the second section explicitly identifies its rules-based fallback. This separation does not verify that historical records are correct or that model-added advice is safe; review still matters.

**Visual before-and-after review:** Upload original and revised PNG/JPEG/WebP screenshots under **Before and after**, create a draft, then expand **Compare visual changes**. Select the matching images and click **Describe visible changes**. This requires a vision-capable configured local model. Edit the proposed observations, confirm you checked both images, and attach them before approving the decision. Each image is limited to 6 MB and 20 megapixels. Reviewed observations stay attached to the decision and are available to the answer writer and Assist Mode. They describe visible edits, not inferred motives or proof of success. For slides or PDF pages, upload screenshots; automatic rendering and video-frame comparison are not included. Tests simulate model responses; real-model visual accuracy remains unmeasured.

**Several decisions from one upload:** After creating a draft, click **Find separate decisions in this work** with the local model connected. Review, interview, approve, or discard each proposed decision individually. Supporting quotations are checked against extracted source text. Approving a decision also preserves original uploaded files in that company's artifact folder; retrieve the decision in Assist Mode and expand **Original work and employee explanation** to download them. Automated tests cover parsing and citation validation; actual model extraction quality depends on the configured model and still needs real-example evaluation.

**Before-and-after capture:** In **Add Company Knowledge → Upload completed work**, choose **Before and after**, upload original and revised work, and optionally add approval emails or test results as supporting evidence. Inspect the extracted-text comparison, answer up to three missing-context questions at a time, then review and approve the decision. Source excerpts and employee-confirmed explanations remain attached to the saved decision. This currently compares text, not slide layout or video frames. See `IMPLEMENTATION_PLAN.md` for the remaining stages.

- **Company access portal:** employees enter a private workspace ID, work email, and password before SkillVault selects any company data. New companies receive isolated profile, user, decision, plan, and search-index storage.
- **Development bypass:** local reviewers can open the original populated Northstar demo directly. Set `SKILLVAULT_ENABLE_DEMO_BYPASS=false` before a real deployment.
- **Assist Mode:** enter a client issue or company task and receive a detailed playbook, source files, similar expert cases, escalation guidance, and general suggestions when the evidence is weak.
- **SkillVault Chat:** have a multi-turn conversation, ask follow-up questions, and inspect the approved company sources used for each answer.
- **SkillVault answer writer:** automatically turns retrieved professional decisions into a detailed playbook, client-response draft, engineering-handoff draft, escalation guidance, and clearly labeled general suggestions.
- **Private local answer engine:** Qwen3.5 runs through Ollama on the same computer or company server. It receives the current question and only the retrieved approved records, then writes a direct adapted response with inline source citations. No OpenAI, Anthropic, or Google model API is used.
- **Current-case adaptation:** every answer synthesizes useful actions from multiple retrieved decisions, maps them to today’s facts, substitutes current scope and values, and still provides at least five useful provisional steps before any escalation guidance when similarity is low.
- **Independent recommendations:** SkillVault clearly separates company-backed actions from broader professional safeguards. When fresh external facts matter, it recommends specific official documentation, release notes, status pages, primary datasets, or regulatory sources to check without sending private case data to a public search engine.
- **Structured recommendation feedback:** employees can rate relevance, clarity, and actionability; identify wrong sources, missing steps, early escalation, unsafe advice, or misunderstood facts; and regenerate the answer with their corrections. Ordinary feedback is logged for review, while only an administrator’s explicit expert approval turns corrected steps into searchable knowledge.
- **Feedback insights:** administrators can review workspace-specific feedback trends, comments, suggested corrections, and expert-approved promotions from Model Insights without exposing employee email addresses in the table.
- **Decision reconstruction from completed work:** PowerPoint decks, code, pull-request diffs, notebooks, tickets, reports, email, and other artifacts become editable records of what happened and why. Anything the artifact cannot establish is shown as missing context and must be confirmed before approval.
- **One-question decision interview:** after extraction, SkillVault asks for the highest-priority missing fact first (such as why a visible change was made), using the observed choice in the question when possible. It never treats an unmeasured outcome as proof of success; the reviewer can record "Not measured yet."
- **Current-case context for workers:** in Chat or Assist Mode, employees can optionally provide their team and upload a current ticket, deck, document, or code file. SkillVault uses bounded extracted text to tailor this answer, without adding the file to approved knowledge. Chat can start directly from the attachment. When a material fact is still missing, it asks one targeted follow-up after giving an initial plan.
- **Decision Lineage (populated competition demo):** traces the current problem through retrieved decisions, retained expert judgment, present-day changes, approval status, and outcome status; approved plans save that lineage with a unique plan ID.
- **Decision ownership and expiration (populated competition demo):** every new decision records an expert owner, department, approval date, last review, expiration date, lifecycle status, and optional replacement source. Expired, outdated, and replaced records remain auditable but are excluded from retrieval.
- **Outcome Review (populated competition demo):** approved plans can be followed through execution by recording whether they worked, what steps changed, and the verified outcome. Successful reviewed results can become new governed company knowledge.
- **Add Expert Decision:** record an expert action after it happened, attach screenshots, diagrams, or screen recordings, confirm it is accurate, and retrain SkillVault immediately.
- **Add Company Knowledge:** choose **Upload completed work** to extract a reviewable decision from real artifacts, or **Import prepared CSV** to approve multiple already-structured decisions at once.
- **Model Insights:** inspect the cases, decision classes, and validation score.

### Scalable knowledge retrieval

Optional semantic search uses a separate local embedding model through [Ollama's embedding endpoint](https://docs.ollama.com/api/embed). Run `ollama pull embeddinggemma`, then open **Model Insights → Build or update semantic search**. Vectors are stored in the company index directory. New or edited decisions are embedded incrementally. Queries combine semantic candidates with keyword matches; keyword search remains available when the model is offline or no semantic index has been built. This version uses exact cosine search over stored vectors, not an approximate nearest-neighbor index. Similarity scores are ranking signals, not calibrated probabilities of correctness. Real-model retrieval quality and large-company performance still require measurement.

SkillVault maintains a persistent local hybrid search index. SQLite FTS5 creates an inverted index over every approved decision and rapidly shortlists relevant records. A persisted TF-IDF vector matrix then reranks only that shortlist before SkillVault builds an evidence-linked answer. The index is reused across Streamlit reruns and application restarts, and it rebuilds automatically only when approved knowledge changes.

This architecture is designed to remain responsive with tens of thousands of decisions without rereading every source document for every question. Retrieval remains local, and only the current question plus shortlisted approved records are passed to the locally hosted model. A production deployment can replace the local index with pgvector, Qdrant, Pinecone, Weaviate, or another managed vector database without changing the approval and answer-writing workflow.

## Run in VS Code

### Optional recording capture

In **Add Company Knowledge → Upload completed work → Explain the decision with a recording**, upload a voice memo or recording with spoken explanation. Click **Transcribe recording locally**, correct any mistakes, and confirm you reviewed the transcript. Only that confirmed transcript enters the draft alongside the other evidence. Approval preserves the original recording and reviewed transcript in the company artifact folder. This transcribes speech; it does not interpret video frames or prove that the employee's explanation is correct.

Install the optional [faster-whisper](https://github.com/SYSTRAN/faster-whisper) dependency and download its speech model once from the project folder:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-speech.txt
.\.venv\Scripts\python.exe -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8')"
```

The download command needs internet access. The app subsequently loads cached speech-model files only and processes recordings locally. The default is `base`; set `SKILLVAULT_SPEECH_MODEL` if you have downloaded a different model. Recording uploads are limited to 50 MB. Automated tests use a simulated transcriber; real-recording accuracy and performance still need evaluation on the intended machine.

Verified source quotations now show their starting slide, PDF page, Word paragraph/table row, or source-code line when available. Scanned PDFs without extractable text still need OCR or an employee explanation; these references do not imply visual analysis.

### Start the app

From the project folder:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

If dependencies have not been installed yet:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The app runs without an API key. To enable the conversational local model on Windows, install Ollama and download Qwen3.5 once:

```powershell
winget install Ollama.Ollama
ollama pull qwen3.5:9b
.\.venv\Scripts\python.exe -m streamlit run app.py
```

`qwen3.5:9b` is about 6.6 GB. On a lower-memory computer, use `ollama pull qwen3.5:4b` and set `SKILLVAULT_LOCAL_MODEL = "qwen3.5:4b"` in `.streamlit/secrets.toml`. The local Ollama API defaults to `http://127.0.0.1:11434` and requires no API key.

See the official [Ollama Windows instructions](https://docs.ollama.com/windows) and [Qwen3.5 9B model page](https://ollama.com/library/qwen3.5:9b) for current installation, hardware, model-size, and license information.

Streamlit Community Cloud cannot run a large local Ollama model inside the normal app container. A real business deployment should run Streamlit and Ollama on the same private server, or configure `SKILLVAULT_OLLAMA_URL` to an access-controlled internal Ollama server. This removes per-answer LLM API charges, but the company still pays for its own CPU/GPU server, storage, and electricity.

### Company workspaces and the separate clean-start version

The main `app.py` now supports multiple isolated local company workspaces. Its access portal can create a new empty workspace or sign an employee into an existing one. The **Development bypass** opens the original populated Northstar Cloud competition demo without changing its data. A second legacy prototype, `blank_app.py`, remains available as a single clean-start workspace.

```powershell
.\.venv\Scripts\python.exe -m streamlit run blank_app.py
```

Main-app company data is stored under a separate opaque workspace ID in `data/local/company_workspaces/`; the private registry is `data/local/company_registry.json`. Clean-start-copy data remains under `data/local/blank_workspace/`. These paths are ignored by Git. Local sign-in uses salted password hashes for prototype testing; production should use company SSO, encrypted managed storage, audit logs, and role-based access controls.

## Sample knowledge pack

The `sample_data/` folder contains the fictional Northstar Cloud source files and `northstar_cloud_cases.csv`. Judges can upload the CSV through **Add Company Knowledge → Import prepared CSV** to test the knowledge-ingestion flow.

Ready-made completed-work examples are also included. Upload `sample_data/completed_work_examples/investor_pitch/Northstar_Cloud_Investor_Pitch.pptx` to capture the decisions behind a completed investor deck. The importer reads both slide content and speaker notes. Additional examples include `output/pdf/northstar_resolved_support_ticket_NC-1842.pdf` and the merged-PR evidence in `sample_data/completed_work_examples/api_code_fix/`.

## Upload format

CSV uploads require `case`, `decision`, and `reasoning`. Recommended decision-context columns are `goal`, `chosen_approach`, `alternatives_considered`, `constraints`, `instructions`, `reusable_rule`, and `exceptions`. Other optional columns are `source_file`, `software`, `methods`, `outcome`, `expert_owner`, `department`, `approval_date`, `last_reviewed_date`, `expiration_date`, `lifecycle_status`, and `replaces_source`. The Add Expert Decision form collects the same fields for one action at a time. This means SkillVault stores the decision *behind* the artifact, not merely the artifact's text.

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

Upload Completed Work extracts text locally from plain text, Markdown, logs, CSV, JSON, source code, diffs, notebooks, email, PDF, Word, and PowerPoint files. PowerPoint speaker notes are included so an employee can preserve the reasoning behind an audience-facing deck. When the local Qwen model is running, uploaded screenshots can also inform the draft; videos remain retained source evidence and need a short note or transcript for important content. Extracted drafts never become retrievable company knowledge until an expert reviews and approves them.

SkillVault passes the current question and shortlisted approved decision records to the local Ollama process, not the entire knowledge base. A real company deployment should still apply data classification, authentication, role-based access control, encrypted storage, retention rules, model-license review, and network isolation before allowing sensitive uploads.
