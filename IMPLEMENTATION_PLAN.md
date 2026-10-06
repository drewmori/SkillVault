# SkillVault: completed work to company assistance

## Stage 1 — capture and interview

Implemented: distinct original/revised/supporting uploads; text comparisons; short missing-context interview; employee-confirmed rationale; source roles, excerpts and fingerprints persisted with approved decisions; evidence available to retrieval and the local answer writer. Completed-work and explanation-only inputs remain supported.

Limitations: comparison currently uses extracted text, not visual layout or video analysis. Long excerpts are bounded with visible warnings. Questions use missing-field rules, while the configured local model structures the evidence.

## Stage 2 — evidence and ingestion quality

- Implemented: local-model splitting into up to six independently reviewed decisions, with exact quotations verified against extracted source text. Sequential review supports approving or discarding each proposal.
- Implemented: original artifacts preserved on approval in company-scoped content-addressed storage; Assist Mode provides downloads alongside source excerpts. Verified quotations identify their starting slide, PDF page, Word paragraph/table row, or code line where available. These are text locators, not visual bounding boxes or full claim ranges.
- Implemented: optional local recording transcription, with editable transcript and explicit employee confirmation before capture. Both the original recording and reviewed transcript are preserved when a decision is approved. Requires a separately installed speech model; real-audio accuracy has not yet been evaluated.
- Implemented: opt-in comparison of an original/revised screenshot pair using the configured local vision model. Employee edits and confirms observations before attaching them to a draft; approved observations remain source-linked and reach the answer writer. PNG/JPEG/WebP images are validated and bounded. Automatic document rendering and video-frame comparison remain to do; real-model visual quality has not been evaluated.
- Validate supported claims and distinguish outcomes that are measured from outcomes still pending.

Validation: automated tests cover source locators, mocked transcription and visual comparison, extraction, and Streamlit interactions. The local Ollama endpoint was unavailable during implementation, so real-model splitting and visual-comparison quality have not yet been evaluated.

## Stage 3 — semantic retrieval

- Implemented: optional local Ollama embeddings alongside lexical search, with incremental company-specific SQLite indexes and exact cosine retrieval. Setup/build controls are in Model Insights. Mocked-vector tests cover paraphrase routing, incremental updates, isolation, and offline fallback; real-model quality and large-library performance remain unmeasured.
- Rank by meaning, context, source validity, and relevant constraints.
- Evaluate paraphrases, distractors, conflicting decisions, stale records, and no-match questions.

## Stage 4 — conversational recommendations

- Implemented: the answer writer receives allowlisted company-profile fields alongside current facts, conversation, and source evidence. Follow-up retrieval includes the last three employee turns. More thorough conversation-state and real-output evaluation remains to do.
- Explain what transfers, what changes, and why; generate useful steps or artifacts.
- Resolve missing facts through follow-up questions and revise the same plan.
- Evaluate real local-model outputs; mocked prompt tests are not answer-quality evidence.

## Stage 5 — pilot readiness

- Test unfamiliar real-world upload → interview → review → search → conversation → feedback journeys.
- Strengthen authentication, authorization across every write path, transactional persistence, and tenant isolation.
- Verify installation, model/hardware availability, and concurrent use before describing the application as production-ready.
