import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const runtime = 'C:/Users/jaymo/.cache/codex-runtimes/codex-primary-runtime/dependencies';
process.env.RUNTIME_NODE_MODULES = runtime + '/node/node_modules';
const skill = 'C:/Users/jaymo/.codex/plugins/cache/openai-primary-runtime/presentations/26.904.11930/skills/presentations';
const {Presentation, PresentationFile} = await import(pathToFileURL(runtime + '/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs'));
const {finalizePresentation} = await import(pathToFileURL(skill + '/container_tools/artifact_tool_utils.mjs'));
const root = process.cwd();
const build = path.join(root, 'output/journey-test/build');
const out = path.join(root, 'output/journey-test/files');
await fs.mkdir(build, {recursive:true});
await fs.mkdir(out, {recursive:true});
for (const version of ['before','after']) {
  const p = Presentation.create({slideSize:{width:1280,height:720}});
  const slide = p.slides.add();
  slide.background.fill = '#F7F8FA';
  const lines = version === 'before' ? [
    'Product architecture',
    'Workflow engine, integrations, and analytics',
    'Customer adoption: [add verified evidence]',
    'Next step: [unfinished]',
  ] : [
    'Customer adoption supports the next pilot',
    '12 pilot stores used the replenishment workflow',
    '9 of 12 stores returned weekly during September',
    'Next step: a 30-day pilot with an agreed success measure',
    'Technical architecture belongs in the appendix',
  ];
  lines.forEach((text,index) => {
    const shape = slide.shapes.add({geometry:'textbox',position:{left:72,top:64+index*105,width:1136,height:90},fill:'none',line:{fill:'none',width:0}});
    shape.text = text;
    shape.text.style = {typeface:'Arial',fontSize:index===0?44:29,bold:index===0,color:'#15283E',autoFit:'none'};
  });
  const foot = slide.shapes.add({geometry:'textbox',position:{left:72,top:645,width:1136,height:40},fill:'none',line:{fill:'none',width:0}});
  foot.text = 'Fictional SkillVault test fixture. All figures are illustrative.';
  foot.text.style = {typeface:'Arial',fontSize:19,color:'#526073'};
  slide.speakerNotes.textFrame.setText('Fictional test artifact. No employee rationale or real-world outcome is supplied in this file.');
  const candidate = path.join(build,version+'.pptx');
  await (await PresentationFile.exportPptx(p)).save(candidate);
  await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath:path.join(out,version+'.pptx'),
    pythonExecutable:runtime+'/python/python.exe',integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
    layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
    layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit'],
    explicitTotalSlideCount:1,fontPolicy:{basis:'design',families:['Arial']},verifyArtifactToolImport:true,
    receiptPath:path.join(build,version+'.validation.json')});
  const preview=await p.export({slide,format:'png',scale:1});
  await fs.writeFile(path.join(out,version+'.png'),new Uint8Array(await preview.arrayBuffer()));
  console.log(version+' fixture exported');
}
