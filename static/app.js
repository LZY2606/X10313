"use strict";
// Macroscope review workbench. No build step; vanilla JS.
const $ = (s, r=document) => r.querySelector(s);
const $$ = (s, r=document) => [...r.querySelectorAll(s)];
const state = {
  projects: [], projectId: null, data: null,
  selectedInput: null, artifacts: [], selectedArtifact: null,
  selectedNode: null, nodeById: {}, resolutions: {}, tokens: [],
  compare: {a:null,b:null},
};

function toast(msg, isErr=false){
  const t=$("#toast"); t.textContent=msg;
  t.style.borderColor=isErr?"#f06b6b":"#5aa7ff";
  t.classList.add("show"); clearTimeout(t._h);
  t._h=setTimeout(()=>t.classList.remove("show"),2600);
}
async function api(path, opts={}){
  const res = await fetch(path,{headers:{"Content-Type":"application/json"},...opts});
  if(!res.ok){ let m=res.statusText; try{ m=(await res.json()).detail||m; }catch{}
    throw new Error(m); }
  return res.json();
}
const esc = s => String(s??"").replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
const short = id => id ? id.replace(/^.*-(n\d+|art_.*|evt_.*)/,(m,g)=>g) : "";

// ---------------------------------------------------------------- init
async function init(){
  bindUI();
  await loadProjects();
}
function bindUI(){
  $("#projectSelect").addEventListener("change",e=>loadProject(e.target.value));
  $("#btnNew").onclick=async()=>{
    const name=prompt("项目名称","我的宏卫生性复核"); if(!name)return;
    const p=await api("/api/projects",{method:"POST",body:JSON.stringify({name})});
    await loadProjects(p.id); toast("已创建项目");
  };
  $("#btnSeed").onclick=async()=>{
    await api("/api/projects/seed",{method:"POST"});
    await loadProjects(); toast("离线案例包已导入并完成基线+反例");
  };
  $("#btnReplay").onclick=onReplay;
  $("#btnExport").onclick=onExport;
  $("#btnWipe").onclick=onWipe;
  $("#fileImport").addEventListener("change",onImportFile);
  $("#btnSubmit").onclick=onSubmitInput;
  $("#btnSample").onclick=fillSample;
  $("#btnCompute").onclick=onCompute;
  $("#btnCompare").onclick=onCompare;
  $("#btnUndo").onclick=onUndo;
  $$(".tab").forEach(b=>b.onclick=()=>{
    $$(".tab").forEach(x=>x.classList.remove("active"));
    $$(".tabpane").forEach(x=>x.classList.remove("active"));
    b.classList.add("active"); $("#tab-"+b.dataset.tab).classList.add("active");
  });
  $$("[data-verdict]").forEach(b=>b.onclick=()=>onDecision(b.dataset.verdict));
}
async function loadProjects(selectId){
  state.projects=await api("/api/projects");
  const sel=$("#projectSelect");
  sel.innerHTML=state.projects.map(p=>`<option value="${p.id}">${esc(p.name)}</option>`).join("")
    || `<option value="">（无项目）</option>`;
  const id=selectId||state.projects[0]?.id||"";
  sel.value=id; if(id) await loadProject(id);
}
async function loadProject(pid){
  state.projectId=pid;
  if(!pid){ state.data=null; render(); return; }
  state.data=await api(`/api/projects/${pid}`);
  state.artifacts=state.data.artifacts;
  state.selectedInput=state.data.inputs.at(-1)?.id||null;
  const cur=state.artifacts.filter(a=>a.input_version_id===state.selectedInput);
  state.selectedArtifact=cur.at(-1)?.id||state.artifacts.at(-1)?.id||null;
  render();
}

// ---------------------------------------------------------------- actions
const SAMPLE = `;; 可复制样例：卫生宏 + 同名局部变量
(define-syntax shadow-x
  (lambda (stx)
    (let ((xs (syntax-e stx)))
      \`(lambda (x) ,(cadr xs)))))
((lambda (x) (shadow-x x)) 11)`;
function fillSample(){ $("#sourceText").value=SAMPLE; }

async function onSubmitInput(){
  const text=$("#sourceText").value;
  const label=$("#newLabel").value||"input.scm";
  if(!text.trim()){ toast("空输入不会被接受",true); return; }
  const parent = state.selectedInput || null;
  try{
    const iv=await api(`/api/projects/${state.projectId}/inputs`,
      {method:"POST",body:JSON.stringify({text,label,parent_version_id:parent})});
    toast(`已提交不可变版本 v${iv.seq_index}（原输入不会被覆盖）`);
    await loadProject(state.projectId);
    state.selectedInput=iv.id; renderVersions();
  }catch(e){ toast(e.message,true); }
}
async function onCompute(){
  if(!state.selectedInput){ toast("请先选择输入版本",true); return; }
  const macro_budget=parseInt($("#budgetValue").value,10);
  const seed=parseInt($("#seedValue").value,10);
  try{
    const art=await api(`/api/projects/${state.projectId}/compute`,{
      method:"POST",body:JSON.stringify({input_version_id:state.selectedInput,
        macro_budget,seed})});
    state.selectedArtifact=art.id;
    await loadProject(state.projectId);
    state.selectedArtifact=art.id; render();
    toast(`展开完成：${art.expansion_status}，身份 ${art.id}`);
  }catch(e){ toast(e.message,true); }
}
async function onDecision(verdict){
  if(!state.selectedNode){ toast("先在展开树/图中选择一个对象",true); return; }
  const note=$("#decisionNote").value||"(无说明)";
  await api(`/api/projects/${state.projectId}/decisions`,{method:"POST",
    body:JSON.stringify({target_type:"node",target_id:state.selectedNode,
      verdict,note,detail:{artifact_id:state.selectedArtifact}})});
  $("#decisionNote").value="";
  await loadProject(state.projectId);
  state.selectedArtifact && keepSelection();
  toast("裁定已写入历史（冲突会标记被取代的决定）");
}
async function onUndo(){
  try{ const r=await api(`/api/projects/${state.projectId}/decisions/undo`,
    {method:"POST"}); await loadProject(state.projectId);
    state.selectedArtifact && keepSelection();
    toast("已撤销："+r.event.summary);
  }catch(e){ toast(e.message,true); }
}
async function onReplay(){
  const r=await api(`/api/projects/${state.projectId}/replay`,{method:"POST"});
  await loadProject(state.projectId);
  toast(r.all_identical?"逐步重放完成：身份/顺序/来源完全一致":"重放存在差异，请检查", !r.all_identical);
}
async function onExport(){
  const d=await api(`/api/projects/${state.projectId}/export`);
  const blob=new Blob([JSON.stringify(d,null,2)],{type:"application/json"});
  const a=document.createElement("a");
  a.href=URL.createObjectURL(blob); a.download=d.project_id+".json"; a.click();
  toast("已导出（含内容哈希，可删除本地库后再导入）");
}
async function onWipe(){
  if(!confirm("删除该项目在本地 SQLite 中的全部数据？可先用导出备份。"))return;
  await api(`/api/projects/${state.projectId}/wipe`,{method:"POST"});
  await loadProjects(); toast("本地项目数据已删除");
}
async function onImportFile(ev){
  const f=ev.target.files[0]; if(!f)return;
  try{
    const body=JSON.parse(await f.text());
    const r=await api("/api/import",{method:"POST",body:JSON.stringify(body)});
    await loadProjects(r.project_id);
    toast("导入完成：身份、顺序与来源关系保持");
  }catch(e){ toast("导入失败："+e.message,true); }
  ev.target.value="";
}
async function onCompare(){
  const a=$("#cmpA").value, b=$("#cmpB").value;
  if(!a||!b){ toast("选择两个展开版本",true); return; }
  const r=await api("/api/compare",{method:"POST",
    body:JSON.stringify({artifact_a:a,artifact_b:b})});
  $("#compareView").textContent=JSON.stringify(r,null,2);
}
function keepSelection(){
  // after reload, reselect artifact then node
  const art=state.artifacts.find(a=>a.id===state.selectedArtifact);
  if(art){ selectArtifact(art.id,false); }
}

// ---------------------------------------------------------------- render
function render(){
  if(!state.data){ blank(); return; }
  renderVersions(); renderArtifactSelectors(); renderSummary();
  renderCurrentArtifact(); renderHistory(); renderArchive();
}
function blank(){
  ["versionList","runSummary","treeView","scopeView","bindingView","mapView",
   "stepsView","graphView","historyView","archiveView","inspector",
   "compareView","sourceTokens"].forEach(id=>$("#"+id).innerHTML="");
}
function currentArtifact(){
  return state.artifacts.find(a=>a.id===state.selectedArtifact)||null;
}
function renderVersions(){
  const list=$("#versionList"); list.innerHTML="";
  const ivs=state.data.inputs;
  ivs.forEach((iv,i)=>{
    const arts=state.artifacts.filter(a=>a.input_version_id===iv.id);
    const stale=arts.some(a=>a.status==="stale") && !arts.some(a=>a.status==="current");
    const active=iv.id===state.selectedInput;
    const parent=iv.parent_version_id
      ? ` ← v${Math.max(0,i-1)}` : (i===0?"（初始版本）":"");
    const div=document.createElement("div");
    div.className=`vitem ${active?"active":""} ${stale?"stale":""}`;
    div.innerHTML=`<div><b>v${iv.seq_index}</b> ${esc(iv.label)}
        <small>${esc(parent)}</small></div>
      <small class="muted">${iv.id}</small>
      <div>${iv.tokens.length} tokens · ${iv.diagnostics.length} 读入诊断
        <span class="tag current">不可变</span>${stale?'<span class="tag stale">派生过期</span>':""}
      </div>`;
    div.onclick=()=>{ state.selectedInput=iv.id;
      const a=state.artifacts.filter(x=>x.input_version_id===iv.id).at(-1);
      if(a)state.selectedArtifact=a.id; render(); };
    list.appendChild(div);
  });
  $("#sourceMeta").textContent=ivs.length?`共 ${ivs.length} 个版本`:"";
}
function renderArtifactSelectors(){
  const opts=state.artifacts.map(a=>{
    const iv=state.data.inputs.find(x=>x.id===a.input_version_id);
    return `<option value="${a.id}" ${a.id===state.selectedArtifact?"selected":""}>
      v${iv?.seq_index??"?"} · ${a.params.macro_budget}预算 · ${a.expansion_status}
      ${a.status==="stale"?"[过期]":""} · ${short(a.id)}</option>`;
  }).join("");
  $("#cmpA").innerHTML=opts; $("#cmpB").innerHTML=opts;
  if(state.artifacts.length>=2){ $("#cmpB").value=state.artifacts.at(-2).id;
    $("#cmpA").value=state.artifacts.at(-1).id; }
  $("#cmpA").onchange=e=>state.compare.a=e.target.value;
  $("#cmpB").onchange=e=>state.compare.b=e.target.value;
}
function selectArtifact(id,doRender=true){
  state.selectedArtifact=id; state.selectedNode=null;
  if(doRender) renderCurrentArtifact();
  $$("#versionList .vitem").forEach(()=>{});
}
function renderSummary(){
  const art=currentArtifact(); const box=$("#runSummary");
  if(!art){ box.innerHTML=`<span class="muted">尚无展开结果</span>`; return; }
  box.innerHTML=`<div class="kv">
    <span class="muted">artifact</span><span>${art.id}</span>
    <span class="muted">算法版本</span><span>${art.algorithm_version}</span>
    <span class="muted">参数指纹</span><span>${art.params_fingerprint}</span>
    <span class="muted">直接来源</span><span>${art.direct_sources.join(", ")}</span>
    <span class="muted">虚拟时间</span><span>${art.created_vt}</span>
    <span class="muted">状态</span><span>${art.status==="stale"
      ?'<span class="tag stale">过期（可查看，不可冒充当前）</span>'
      :'<span class="tag current">current</span>'}</span>
    <span class="muted">展开</span><span>${art.expansion_status}
      （宏调用 ${art.call_count}/${art.macro_budget}）</span>
  </div>`;
}

function renderCurrentArtifact(){
  const art=currentArtifact();
  if(!art){ ["treeView","scopeView","bindingView","mapView","stepsView",
             "graphView","sourceTokens"].forEach(id=>$("#"+id).innerHTML="");
    $("#inspector").textContent="未选中节点"; return; }
  const graph=art.graph;
  state.nodeById=Object.fromEntries(graph.nodes.map(n=>[n.id,n]));
  state.resolutions=art.resolutions||{};
  state.tokens=art.tokens||[];
  renderTokens(art);
  renderTree(art);
  renderScopes(art);
  renderBindings(art);
  renderSourceMap(art);
  renderSteps(art);
  renderGraphSvg(art);
}
function renderTokens(art){
  const box=$("#sourceTokens");
  box.innerHTML=art.tokens.map(t=>{
    const d=(art.reader_diagnostics||[]).find(x=>x.span&&
      x.span.offset===t.span.offset);
    return `<span class="tok ${d?(d.severity==="error"?"err":"warn"):''}"
      title="tok${t.index} · L${t.span.line}:C${t.span.col} · ${t.span.unit}">
      ${esc(t.value===" "?",":t.value)}</span>`;
  }).join("");
}
function nodeFlags(n){ return (n.flags||[]).map(f=>
  `<span class="sflag ${f}">${f}</span>`).join(""); }
function renderTree(art){
  const rootId=art.expanded_root_id;
  const root=state.nodeById[rootId];
  if(!root){ $("#treeView").innerHTML=`<span class="diag error">无展开根</span>`; return; }
  $("#treeView").innerHTML=`<ul>${treeNode(root)}</ul>`;
  $$("#treeView .node-line").forEach(el=>el.onclick=()=>
    selectNode(el.dataset.id));
}
function treeNode(n){
  const r=state.resolutions[n.id];
  const rstat=r?` <small class="muted">[${r.status}]</small>`:"";
  const cls=["node-line"];
  if(n.flags?.includes("explicit-capture"))cls.push("explicit-capture");
  if(n.flags?.some(f=>f.includes("ambiguous")))cls.push("ambiguous");
  if(n.flags?.some(f=>f==="budget-exhausted"||f==="macro-error"))cls.push("budget");
  if(n.id===state.selectedNode)cls.push("sel");
  const label=n.kind==="seq"
    ? (n.bracket==="{}"?"{…}":"(…)")
    : `<span class="sym">${esc(n.lexeme)}</span>`;
  const head=`<li><span class="${cls.join(" ")}" data-id="${n.id}">
      <span class="nid">${short(n.id)}</span>${label}${rstat}${nodeFlags(n)}
    </span>`;
  if(n.items?.length)
    return head+`<ul>${n.items.map(id=>treeNode(state.nodeById[id])).join("")}</ul></li>`;
  return head+"</li>";
}
function scopeChips(scopes){
  return (scopes||[]).map(s=>{
    const pre=s[0];
    return `<span class="scopechip ${pre}">${esc(s)}</span>`;
  }).join("");
}
function renderScopes(art){
  const rows=Object.values(state.nodeById).filter(n=>n.kind==="id")
    .map(n=>{ const r=state.resolutions[n.id];
      return `<tr><td>${esc(n.value)} <small class="muted">${short(n.id)}</small></td>
      <td>${scopeChips(n.scopes)}</td>
      <td>${r?r.status:"-"}</td>
      <td>${r?.binding?esc(r.binding.kind+" @ "+short(r.binding.site)):""}</td></tr>`;
    }).join("");
  $("#scopeView").innerHTML=`<table><thead><tr><th>标识符</th><th>scope 集</th>
    <th>判定</th><th>绑定</th></tr></thead><tbody>${rows}</tbody></table>
  <p class="muted">捕获完全由 scope 集合的子集关系判定（g 模块 / b 绑定 /
    m 宏标记 / i 引入），不做字符串改名。</p>`;
}
function renderBindings(art){
  const bs=art.bindings||[];
  const rows=bs.map(b=>`<tr>
    <td>${esc(b.symbol)}</td><td>${esc(b.kind)}</td>
    <td>${scopeChips(b.binder_scopes)}</td>
    <td><a data-node="${b.site}" class="bindlink">${short(b.site)}</a></td>
    <td>${esc(b.detail||"")}</td></tr>`).join("");
  // candidate rows for ambiguous / resolved ids
  const cand=Object.values(art.resolutions||{}).flatMap(r=>
    (r.candidates||[]).map(c=>`<tr>
      <td>${esc(r.symbol)} <small>${short(r.node_id)}</small></td>
      <td>${esc(c.kind)}</td>
      <td class="${c.maximal?"cmp-add":""}">${c.specificity}</td>
      <td>${c.eligible?"✓":"✗"} ${c.maximal?"◀":""}</td>
      <td>${esc(c.reason)}</td></tr>`));
  $("#bindingView").innerHTML=`<table><thead><tr><th>符号</th><th>binder</th>
    <th>binder scope 集</th><th>定义点</th><th>说明</th></tr></thead>
    <tbody>${rows}</tbody></table>
    <h3>全部合法候选（含拒绝原因/特异性）</h3>
    <table><thead><tr><th>use</th><th>候选</th><th>特异性</th><th>入选</th>
    <th>理由</th></tr></thead><tbody>${cand.join("")}</tbody></table>`;
  $$("#bindingView .bindlink").forEach(a=>a.onclick=()=>selectNode(a.dataset.node));
}
function renderSourceMap(art){
  const rows=Object.values(state.nodeById)
    .filter(n=>n.span&&n.kind!=="seq").map(n=>`
    <tr><td><a data-node="${n.id}" class="bindlink">${short(n.id)}</a></td>
    <td>${esc(n.kind)}</td><td>${esc(n.lexeme)}</td>
    <td>L${n.span.line}:C${n.span.col} (offset ${n.span.offset}+${n.span.length})</td>
    <td>${n.parents.map(p=>`<a data-node="${p}" class="bindlink">${short(p)}</a>`).join(" ")||"—"}</td>
    <td>${esc(n.reason)}</td></tr>`).join("");
  $("#mapView").innerHTML=`<div class="muted">${esc(art.tokens?.[0]?.span?.unit||"")}</div>
  <table><thead><tr><th>节点</th><th>kind</th><th>词素</th><th>源 span</th>
  <th>直接来源</th><th>转换</th></tr></thead><tbody>${rows}</tbody></table>`;
  $$("#mapView .bindlink").forEach(a=>a.onclick=()=>selectNode(a.dataset.node));
}
function renderSteps(art){
  const steps=(art.graph.steps||[]).map(s=>{
    const keys=Object.entries(s).filter(([k])=>!["seq","kind"].includes(k))
      .map(([k,v])=>{const vs=typeof v==="object"?JSON.stringify(v):v;
        return `<span class="muted">${k}</span>=${esc(String(v))}`;}).join(" · ");
    return `<div class="step" data-node="${s.node||s.input_node||s.result_node||s.context_node||""}">
      <span class="seq">#${s.seq}</span><b>${esc(s.kind)}</b> <small>${keys}</small></div>`;
  }).join("");
  const gens=(art.generated_symbols||[]).map(g=>
    `<div class="stackframe">生成符号 <b>${esc(g.symbol)}</b>
      · intro scope <span class="scopechip i">${esc(g.intro_scope)}</span>
      · 宏 ${esc(g.macro)} #${g.call} · node <a class="bindlink"
      data-node="${g.node}">${short(g.node)}</a></div>`).join("");
  $("#stepsView").innerHTML=`<div class="muted">宏调用栈与生成符号台账</div>${gens}${steps}`;
  $$("#stepsView .bindlink").forEach(a=>a.onclick=()=>selectNode(a.dataset.node));
  $$("#stepsView .step").forEach(el=>el.onclick=()=>{ if(el.dataset.node) selectNode(el.dataset.node);});
}

// ---------------------------------------------------------------- SVG graph
function renderGraphSvg(art){
  const nodes=art.graph.nodes, edges=art.graph.edges;
  const rootId=art.expanded_root_id;
  // layered layout by depth from expanded root; fallback: node order
  const depth={};
  (function walk(id,d){ if(depth[id]!==undefined&&depth[id]>=d)return;
    depth[id]=Math.max(depth[id]??0,d);
    const n=state.nodeById[id]; (n?.items||[]).forEach(c=>walk(c,d+1)); })(rootId,0);
  // account for derive edges from read roots: place orphans at depth 0
  nodes.forEach(n=>{ if(depth[n.id]===undefined) depth[n.id]=0; });
  const byLevel={};
  nodes.forEach(n=>{ const d=depth[n.id]??0; (byLevel[d]=byLevel[d]||[]).push(n); });
  const levels=Object.keys(byLevel).map(Number).sort((a,b)=>a-b);
  const W=120, H=44, GAPX=28, GAPY=70, PAD=30;
  const pos={}; let maxRows=0;
  levels.forEach(d=>{
    const arr=byLevel[d]; maxRows=Math.max(maxRows,arr.length);
    arr.forEach((n,i)=>{ pos[n.id]={x:PAD+i*(W+GAPX), y:PAD+d*(H+GAPY)}; });
  });
  const width=Math.max(PAD*2+W, PAD*2+maxRows*(W+GAPX));
  const height=PAD*2+(levels.length)*(H+GAPY);
  const edgeSvg=edges.map(e=>{
    const a=pos[e.src], c=pos[e.dst]; if(!a||!c)return"";
    return `<path class="gedge ${e.kind}" d="M${a.x+W/2},${a.y+H}
      C${a.x+W/2},${(a.y+c.y)/2} ${c.x+W/2},${(a.y+c.y)/2}
      ${c.x+W/2},${c.y}"><title>${esc(e.detail)}</title></path>`;
  }).join("");
  const nodeSvg=nodes.map(n=>{
    const p=pos[n.id];
    const flagCls=(n.flags||[]).map(f=>f.includes("capture")?"explicit-capture":
      f.includes("ambiguous")?"ambiguous":
      (f==="budget-exhausted"||f==="macro-error")?"budget":"")
      .filter(Boolean).join(" ");
    const sel=n.id===state.selectedNode?"sel":"";
    const label=n.kind==="seq"?(n.reason.replace(/^expand-/,"").slice(0,14)):
      (n.lexeme||n.kind);
    const fill=n.kind==="seq"?"#16202b":"#1b2735";
    return `<g class="gnode ${flagCls} ${sel}" data-id="${n.id}"
       transform="translate(${p.x},${p.y})">
      <rect width="${W}" height="${H}" rx="7" fill="${fill}"/>
      <text x="6" y="16" class="gnid">${short(n.id)}</text>
      <text x="6" y="33">${esc(String(label).slice(0,15))}</text></g>`;
  }).join("");
  $("#graphView").innerHTML=
    `<svg width="${width}" height="${height}" id="macroSvg">
      <defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3"
        orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#3a4757"/></marker></defs>
      ${edgeSvg}${nodeSvg}</svg>`;
  $$("#graphView .gnode").forEach(g=>g.onclick=()=>selectNode(g.dataset.id));
}
function selectNode(id){
  if(!state.nodeById[id])return;
  state.selectedNode=id;
  renderTree(currentArtifact()); renderGraphSvg(currentArtifact());
  renderInspector();
  highlightSource(id);
}
function renderInspector(){
  const n=state.nodeById[state.selectedNode];
  if(!n){ $("#inspector").textContent="未选中节点"; return; }
  const r=state.resolutions[n.id];
  const art=currentArtifact();
  const incoming=art.graph.edges.filter(e=>e.dst===n.id);
  const outgoing=art.graph.edges.filter(e=>e.src===n.id);
  $("#inspector").textContent=JSON.stringify({
    id:n.id, kind:n.kind, value:n.value, lexeme:n.lexeme,
    flags:n.flags, scopes:n.scopes,
    span:n.span, parents:n.parents, reason:n.reason,
    items:n.items,
    resolution:r||null,
    incoming_edges:incoming.map(e=>({id:e.id,from:e.src,detail:e.detail})),
    outgoing_edges:outgoing.map(e=>({id:e.id,to:e.dst,detail:e.detail})),
    provenance:`${n.reason} ← ${n.parents.join(", ")||"(源 token)"}`,
  },null,2);
  // switch to tree tab and scroll node into view there
}
function highlightSource(id){
  const n=state.nodeById[id];
  $$("#sourceTokens .tok").forEach(t=>t.classList.remove("source-hl"));
  if(!n?.span)return;
  const tok=state.tokens.find(t=>t.span.offset===n.span.offset);
  if(tok){ const el=$$("#sourceTokens .tok")[tok.index];
    el?.classList.add("source-hl"); el?.scrollIntoView({block:"nearest"}); }
}

// ---------------------------------------------------------------- history
function renderHistory(){
  const evs=state.data.events||[];
  const decs=state.data.decisions||[];
  const decByEvent=Object.fromEntries(decs.map(d=>[d.event_id,d]));
  $("#historyView").innerHTML=evs.map(e=>{
    const d=decByEvent[e.id];
    const undone=e.undo_of?"undo":(d?.superseded_by?"stale":"");
    const tag=e.kind.startsWith("decision-accept")?'<span class="verd true">接受</span>':
      e.kind.startsWith("decision-reject")?'<span class="verd false">否定</span>':
      e.kind.startsWith("decision-adjust")?'<span class="tag param">调参</span>':
      e.kind==="decision-undone"?'<span class="tag stale">撤销</span>':"";
    const sup=d?.superseded_by?`<div class="muted">${
      d.superseded_by==="<undone>"?"已撤销":`被 ${short(d.superseded_by)} 取代（冲突保留）`}</div>`:"";
    return `<div class="hitem ${undone}"><span class="vt">#${e.seq} ${e.vt}</span>
      ${tag} ${esc(e.summary)}${sup}
      <details><summary class="muted">机器事件</summary><pre>${esc(JSON.stringify(e.detail,null,2))}</pre></details>
      </div>`;
  }).reverse().join("");
}
function renderArchive(){
  const exps=state.data.experiments||[];
  $("#archiveView").innerHTML=exps.map(e=>{
    const f=e.findings?.[0]||{};
    const mut=state.artifacts.find(a=>a.id===e.mutated_artifact_id);
    const base=state.artifacts.find(a=>a.id===e.baseline_artifact_id);
    return `<div class="aitem" data-base="${e.baseline_artifact_id}"
        data-mut="${e.mutinated_artifact_id||""}">
      <div><b>${esc(e.title)}</b>
        <span class="verd ${e.adopted}">${e.adopted?"采纳":"未采纳"}</span>
        <span class="verd ${f.holds}">判定 ${f.holds?"成立":"不成立"}</span></div>
      <div class="muted">${esc(e.scenario)} · 局部影响 ${
        e.locality_ok===null?"-":e.locality_ok?"已正确限制":"越界!"}</div>
      <div class="muted">基线 ${short(e.baseline_artifact_id)} → 修订 ${
        short(e.mutated_artifact_id)}</div>
      <details><summary>候选 / 拒绝原因 / 证据</summary>
        <pre class="inspector smallp">${esc(JSON.stringify({
          candidate_outcomes:e.candidate_outcomes,
          findings:f, affected:e.affected, note:e.decision_note},null,2))}</pre>
      </details></div>`;
  }).join("")||`<span class="muted">载入案例包后这里出现反例档案</span>`;
  $$("#archiveView .aitem").forEach(el=>el.onclick=ev=>{
    if(ev.target.tagName==="SUMMARY"||ev.target.tagName==="PRE")return;
    const mut=el.dataset.mut||el.dataset.base;
    if(mut){ selectArtifact(mut); render(); }
  });
}

init().catch(e=>toast(String(e),true));
