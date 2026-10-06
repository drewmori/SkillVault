"""Real PPTX -> app backend -> isolated approval -> retrieval acceptance exercise.

No simulated LLM outputs. Writes a viewable report, never production knowledge.
Run from the repository root: python tests/run_upload_journey.py.
Returns a nonzero exit status when an acceptance check fails.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import json
import html
import re
import os
from dataclasses import asdict, replace
from tempfile import TemporaryDirectory
from time import perf_counter
from datetime import datetime, timezone

from skillvault.capture import capture_work, apply_interview, interview_questions
from skillvault.importer import WorkFile, extract_text
from skillvault.data import ExpertCase, infer_features, build_demo_cases
from skillvault.engine import SkillVaultEngine
from skillvault.storage import save_cases, load_cases, append_case, save_recommendation_feedback, load_recommendation_feedback, save_approved_plan, load_approved_plans, save_plan_outcome
from skillvault.artifacts import store_artifacts, read_artifact
from skillvault.local_model import SkillVaultLocalModel, LocalModelError
from skillvault.feedback_learning import learning_profile
from hashlib import sha256


def main():
    out = ROOT / 'output/journey-test'
    files = out / 'files'
    before = WorkFile('before.pptx', 'application/vnd.openxmlformats-officedocument.presentationml.presentation', (files / 'before.pptx').read_bytes())
    after = WorkFile('after.pptx', before.content_type, (files / 'after.pptx').read_bytes())
    checks = []
    def check(name, condition, detail=''):
        checks.append(dict(name=name, status='PASS' if condition else 'FAIL', detail=detail))
    backend = SkillVaultLocalModel(model=os.environ.get('SKILLVAULT_JOURNEY_MODEL', 'qwen3.5:9b'))
    status = backend.status(refresh=True)
    initial = capture_work([before], [after], [])
    extracted = {file.name: extract_text(file)[0] for file in (before, after)}
    check('Real PowerPoint text extracted', '12 pilot stores' in extracted['after.pptx'] and '[unfinished]' in extracted['before.pptx'])
    check('Before/after changes detected', '+12 pilot stores' in initial.capture_evidence['comparison'])
    missing = interview_questions(initial, 20)
    check('Missing employee rationale prompts a question', 'reasoning' in [key for key, _ in missing], str(missing))
    check('Unknown outcome prompts an outcome question', 'outcome' in [key for key, _ in missing], 'The file explicitly supplies no measured outcome. Extracted outcome: ' + initial.outcome)
    model_extraction = None
    model_error = ''
    if status.available:
        try:
            model_extraction = backend.enrich_decision_draft(initial, 'Investor presentation')
        except LocalModelError as error:
            model_error = str(error)
        check('Real model produces a decision draft', model_extraction is not None, model_error)
        if model_extraction is not None:
            check('Real model retains uncertainty about missing rationale',
                  not model_extraction.reasoning.strip() or 'expert must add' in model_extraction.reasoning.lower()
                  or any(any(word in item.lower() for word in ('reason', 'why', 'rationale')) for item in model_extraction.missing_context),
                  'Inspect the full extracted draft in the report for invented motives.')
    # Explicit fixture employee testimony, NOT inferred by the model.
    testimony = {
        'goal': 'Prepare a short investor presentation for the replenishment pilot',
        'chosen_approach': 'Lead with verified pilot adoption and move architecture to the appendix',
        'reasoning': 'The investor audience needed proof that stores repeatedly used replenishment, rather than an architecture tour.',
        'instructions': '1. Check the pilot usage export and date range.\n2. State the number of pilot stores and how many returned weekly.\n3. Lead the opening slide with those verified adoption figures.\n4. Move technical architecture to the appendix.\n5. Close with a 30-day pilot and an agreed success measure.',
        'outcome': 'The slide revision is complete. Investor response has not been measured.',
        'alternatives_considered': 'Opening with architecture was rejected for this nontechnical investor audience.',
        'constraints': 'Eight-minute presentation. Use verified figures only. All numbers in this test are fictional.',
        'exceptions': 'A technical diligence meeting may need architecture first.',
    }
    reviewed = apply_interview(model_extraction or initial, testimony)
    reviewed.summary = 'Investor presentation for a retail replenishment pilot: adoption evidence before architecture'
    case = ExpertCase(reviewed.summary, reviewed.decision, reviewed.reasoning, infer_features(reviewed.summary),
                      source_file=reviewed.source_file, instructions=reviewed.instructions, outcome=reviewed.outcome,
                      goal=reviewed.goal, chosen_approach=reviewed.chosen_approach, constraints=reviewed.constraints,
                      alternatives_considered=reviewed.alternatives_considered, exceptions=reviewed.exceptions,
                      expert_owner='Fictional test reviewer', capture_evidence=reviewed.capture_evidence)
    results = []
    with TemporaryDirectory(prefix='skillvault-journey-') as folder:
        root = Path(folder)
        path = root / 'company-a' / 'cases.json'
        path.parent.mkdir()
        distractors = build_demo_cases()
        save_cases(path, distractors)
        check('Draft does not automatically enter approved knowledge', all(c.source_file != case.source_file for c in load_cases(path)))
        append_case(path, case)
        restored = load_cases(path)
        check('Reviewed decision survives saving and reloading', restored[-1].capture_evidence == case.capture_evidence and restored[-1].reasoning == testimony['reasoning'])
        store_artifacts(root / 'company-a' / 'artifacts', [before, after])
        check('Original PowerPoint preserved exactly', read_artifact(root / 'company-a' / 'artifacts', sha256(after.data).hexdigest()) == after.data)
        check('Other company cannot retrieve original artifact', read_artifact(root / 'company-b' / 'artifacts', sha256(after.data).hexdigest()) is None)
        engine = SkillVaultEngine(restored, index_path=root / 'company-a' / 'index')
        engine.train()
        reopened = SkillVaultEngine(load_cases(path), index_path=root / 'company-a' / 'index')
        reopened.train()
        check('Persistent search index reused after restart', not reopened.index_rebuilt)
        questions = [
            'How should I structure an investor presentation for our replenishment pilot and show weekly store adoption?',
            'Tomorrow I pitch investors about replenishment. I have five minutes instead of eight. Should I start with system architecture or proof stores keep using it?',
            'How do I repair the steam valve inside an espresso machine?',
        ]
        for index, question in enumerate(questions):
            start = perf_counter()
            prediction = reopened.predict(question)
            milliseconds = round((perf_counter() - start) * 1000, 2)
            source_rank = next((rank for rank, hit in enumerate(prediction.similar_cases, 1) if hit.source_file == case.source_file), None)
            answer = reopened.compose_response(question, prediction)
            model_answer = None
            if status.available:
                try:
                    model_answer = backend.grounded_answer(question, prediction)
                except LocalModelError as error:
                    model_error = str(error)
                check(f'Real model answers question {index+1}', bool(model_answer), model_error)
                if model_answer and index == 1:
                    check('Real model addresses five-minute constraint', bool(re.search(r'five.minute|5.minute', model_answer, re.I)))
                if model_answer and index == 2:
                    adapted_section = model_answer.split('## 2.', 1)[-1]
                    check('Real model avoids unrelated client-meeting rationale', 'Send a concise written summary first' not in adapted_section)
                    check('Real model cites no unsupported company source', '[Source:' not in adapted_section)
            if index < 2:
                check(f'Related question {index+1} retrieves uploaded decision in top five', source_rank is not None, f'rank={source_rank}, top match={prediction.knowledge_match:.3f}')
            else:
                check('Unrelated question marked weak rather than confident', prediction.knowledge_match < .35, f'match={prediction.knowledge_match:.3f}')
            check(f'Question {index+1} has both recommendation sections', '## 1. Company-data-only' in answer and '## 2. Adapted recommendation' in answer)
            if index == 1:
                plan = answer.split('#### Adapted action plan for this request', 1)[-1].split('#### Independent recommendations', 1)[0]
                check('Answer adapts the plan to the new five-minute limit', bool(re.search(r'five.minute|5.minute|300.second|\d+\s*(?:seconds|minutes)\b', plan, re.I)), 'Requires explicit timing in the action plan, not merely repeating the question or saying actual time limit.')
            if index == 2:
                check('Unrelated historical reasoning excluded from adapted recommendation', 'Send a concise written summary first' not in answer.split('## 2.', 1)[-1], 'A client-meeting decision should not be offered as reasoning for mechanical repair.')
            results.append(dict(question=question, match=prediction.knowledge_match, uploaded_decision_rank=source_rank,
                                retrieval_ms=milliseconds, search_mode=prediction.retrieval_mode,
                                top_sources=[dict(file=hit.source_file, similarity=hit.similarity) for hit in prediction.similar_cases],
                                fallback_answer=answer, real_model_answer=model_answer))
        feedback_path = root / 'company-a' / 'feedback.json'
        save_recommendation_feedback(feedback_path, dict(workspace_id='a', learn_for_future=True, query=questions[0], verdict='Needs improvement', issue_types=['Too short','Too vague'], reviewer={'user_id':'test'}))
        profile = learning_profile(load_recommendation_feedback(feedback_path), questions[0], 'a')
        prediction = reopened.predict(questions[0])
        adapted = reopened.compose_response(questions[0], prediction, feedback_profile=profile)
        check('Saved feedback changes a future fallback answer', adapted != results[0]['fallback_answer'])
        check('Feedback does not rewrite company-only evidence', adapted.split('## 2.')[0] == results[0]['fallback_answer'].split('## 2.')[0])
        check('Feedback is isolated from company B', not learning_profile(load_recommendation_feedback(feedback_path), questions[0], 'b')['rules'])
        plan_path = root / 'company-a' / 'plans.json'
        plan_id = save_approved_plan(plan_path, questions[0], prediction.label, prediction.confidence, prediction.sources, adapted, reopened.build_decision_lineage(questions[0], prediction))
        check('Approved plan saves its answer and lineage', load_approved_plans(plan_path)[0]['answer'] == adapted)
        save_plan_outcome(plan_path, plan_id, 'Worked', 'Shortened the opening', 'Fictional reviewer approved the revised slide', False)
        check('Outcome follow-up survives reload without automatic knowledge promotion', load_approved_plans(plan_path)[0]['outcome_feedback']['promoted_to_knowledge'] is False and len(load_cases(path)) == len(restored))
    report = dict(timestamp=datetime.now(timezone.utc).isoformat(), checks=checks, total_search_records=len(restored),
                  ai_status=asdict(status), ai_error=model_error, extracted_text=extracted,
                  initial_draft=asdict(initial), real_model_extraction=asdict(model_extraction) if model_extraction else None,
                  employee_testimony=testimony, reviewed_decision=asdict(reviewed), queries=results,
                  feedback_profile=profile, answer_after_feedback=adapted,
                  limitations=['No browser file-upload click journey was performed by this script. It passes actual PPTX bytes through the same backend used by the app.',
                               'Approval and employee testimony are explicit test actions, not autonomous AI inference.',
                               'Absent local model means generative interpretation and answer quality cannot be evaluated.',
                               'Automated assertions do not prove correctness of arbitrary answers or production readiness.'])
    (out / 'results.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    esc = html.escape
    body = '<h1>SkillVault upload-to-answer test</h1><p>Actual execution results. Fictional test company and data. Production knowledge was not modified.</p>'
    passed = sum(c['status'] == 'PASS' for c in checks)
    body += f'<h2>{passed}/{len(checks)} journey checks passed</h2><p>Failures below identify real prototype limitations. They are not hidden by the passing regression suite.</p>'
    body += '<h2>Model availability</h2><pre>' + esc(json.dumps(asdict(status), indent=2)) + '</pre>'
    body += '<p><strong>The separate regression suite is run with <code>python -m unittest discover -s tests -q</code>. This report adds a real-file backend journey.</strong></p>'
    body += '<h2>Original and finished slide</h2><img src="files/before.png"><img src="files/after.png">'
    body += '<h2>Checks</h2><ul>' + ''.join('<li><strong>'+c['status']+'</strong> '+esc(c['name'])+' '+esc(c['detail'])+'</li>' for c in checks) + '</ul>'
    body += '<h2>What the extractor actually produced before employee help</h2><pre>'+esc(json.dumps(asdict(initial), indent=2, default=str))+'</pre>'
    body += '<h2>Employee explanation supplied explicitly in the test</h2><pre>'+esc(json.dumps(testimony, indent=2))+'</pre>'
    for item in results:
        body += '<h2>'+esc(item['question'])+'</h2><p>Uploaded record rank: '+str(item['uploaded_decision_rank'])+'; top similarity: '+str(round(item['match']*100,1))+'%; retrieval: '+str(item['retrieval_ms'])+' ms</p>'
        body += '<h3>Actual rules-based answer</h3><pre>'+esc(item['fallback_answer'])+'</pre>'
        body += '<h3>Real local-model answer</h3><pre>'+esc(item['real_model_answer'] or 'NOT RUN: local model unavailable')+'</pre>'
    body += '<h2>Limitations</h2><ul>'+''.join('<li>'+esc(x)+'</li>' for x in report['limitations'])+'</ul>'
    (out / 'report.html').write_text('<!doctype html><meta charset="utf-8"><title>SkillVault test results</title><style>body{font:17px system-ui;max-width:1100px;margin:40px auto;padding:20px;color:#172b40;background:#f8fafc}pre{white-space:pre-wrap;background:white;padding:20px;border:1px solid #ddd;font-size:14px}img{width:49%}li{margin:10px 0}h2{margin-top:40px}</style>'+body, encoding='utf-8')
    print(json.dumps({key:report[key] for key in ('checks','total_search_records','ai_status')}, indent=2))
    print('REPORT:', out / 'report.html')
    return 0 if passed == len(checks) else 1


if __name__ == '__main__':
    raise SystemExit(main())
