"""Build a self-contained HTML visualization of the whole dataset.

One page, six tabs — the stages have different schemas on purpose, so the
viewer is a tabbed master-detail UI joined by the pipeline's natural keys
(action_type, state name, probe_id) rather than a forced common table:

  Overview      counts, decision matrix, label/pair distributions
  Grounded      probe results: label badge, evidence, undo trace, screenshots
  Trajectories  S2 rollouts: step table, axtree snapshots, screenshots
  States        S3/scale risk-affording states with the risky control
  SFT           filterable samples; tagged assistant sequence rendered
  DPO           chosen vs rejected side by side, pair-type badges

Self-contained by design (like the Qwen-AgentWorld demo): JSON + JPEG
thumbnails are embedded, the file opens from disk with no server and no
external requests. Pure stdlib; Pillow is optional (screenshot thumbnails
are skipped without it).
"""
from __future__ import annotations

import base64
import collections
import io
import json
from datetime import datetime, timezone
from pathlib import Path

from .. import config


# --------------------------------------------------------------------------- #
# loaders
# --------------------------------------------------------------------------- #
def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for ln in path.open(encoding="utf-8"):
        ln = ln.strip()
        if ln:
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
    return out


def _thumb_b64(path: Path, max_w: int = 560, quality: int = 68) -> str:
    """JPEG thumbnail as a data URI; '' when Pillow is missing or file bad."""
    try:
        from PIL import Image

        img = Image.open(path).convert("RGB")
        if img.width > max_w:
            img = img.resize((max_w, int(img.height * max_w / img.width)))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=quality)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


def _clip(text: str, n: int) -> str:
    text = text or ""
    return text if len(text) <= n else text[:n] + f"\n… (+{len(text) - n} chars)"


def _goal_obs(user_content: str) -> tuple[str, str]:
    """Split the user turn into (goal, observation) via the shared parser
    (handles both the 3-section P0+ format and legacy 2-section rows)."""
    from .. import prompts

    p = prompts.parse_user(user_content)
    return p["goal"], p["obs"]


# --------------------------------------------------------------------------- #
# payload assembly
# --------------------------------------------------------------------------- #
def collect_payload(data_root: Path | None = None, max_shots: int = 60) -> dict:
    root = data_root or config.DATA_ROOT
    shots_budget = {"n": max_shots}

    def take_shot(rel: str) -> str:
        if not rel or shots_budget["n"] <= 0:
            return ""
        p = root / rel
        if not p.exists():
            return ""
        b64 = _thumb_b64(p)
        if b64:
            shots_budget["n"] -= 1
        return b64

    # -- grounded probes ---------------------------------------------------- #
    grounded_rows = _jsonl(root / "grounded" / "reversibility.jsonl")
    manifest = _jsonl(root / "grounded" / "MANIFEST.jsonl")
    latest: dict[str, dict] = {}
    for r in grounded_rows:               # effective label = latest non-UNKNOWN
        at = r.get("action_type", "?")
        if r.get("label") != "UNKNOWN" or at not in latest:
            latest[at] = r
    grounded = []
    for r in grounded_rows:
        ev = dict(r.get("evidence") or {})
        shots = []
        sdir = ev.pop("screenshots_dir", "")
        if sdir and (root / sdir).is_dir():
            for p in sorted((root / sdir).glob("*.png"))[:8]:
                b64 = take_shot(str(p.relative_to(root)))
                if b64:
                    shots.append({"name": p.stem, "b64": b64})
        grounded.append({
            "action_type": r.get("action_type", "?"), "label": r.get("label", "?"),
            "grounding": r.get("grounding", ""), "destructive": r.get("destructive", False),
            "commit_mode": r.get("commit_mode", False), "probe_id": r.get("probe_id", ""),
            "timestamp": r.get("timestamp", ""), "probe_name": r.get("probe_name", ""),
            "evidence": ev, "undo_actions": (r.get("evidence") or {}).get("undo_actions", []),
            "effective": latest.get(r.get("action_type")) is r, "shots": shots,
        })

    # -- trajectories -------------------------------------------------------- #
    meta = _jsonl(root / "raw" / "trajectories_meta.jsonl")
    meta_by_id = {m.get("trajectory_id"): m for m in meta}
    trajectories = []
    traj_dir = root / "raw" / "trajectories"
    for f in sorted(traj_dir.glob("*.jsonl")) if traj_dir.exists() else []:
        steps_raw = _jsonl(f)
        if not steps_raw:
            continue
        tid = steps_raw[0].get("trajectory_id", f.stem)
        m = meta_by_id.get(tid, {})
        steps = []
        for s in steps_raw:
            steps.append({
                "i": s.get("step_id"), "action": s.get("action", ""),
                "url": s.get("url_after", ""), "reward": s.get("reward", 0),
                "axtree": _clip(s.get("obs_after_axtree", ""), 1600),
                "shot": take_shot(s.get("screenshot", "")),
            })
        trajectories.append({
            "id": tid, "task_id": steps_raw[0].get("task_id", ""),
            "success": bool(m.get("success")), "max_reward": m.get("max_reward", 0),
            "n_steps": len(steps), "final": (m.get("final_model_response") or "")[:300],
            "steps": steps,
        })

    # -- states --------------------------------------------------------------#
    states = []
    for fname, source in [("pilot_reached_states.jsonl", "pilot"),
                          ("scaled_reached_states.jsonl", "scaled")]:
        for r in _jsonl(root / "raw" / "state_bank" / fname):
            if not r.get("reached"):
                continue
            states.append({
                "name": r.get("name", ""), "action_type": r.get("action_type", ""),
                "url": r.get("url", ""), "source": source,
                "risky": (r.get("risky_action") or {}).get("text", ""),
                "safe_answer": r.get("safe_answer", ""),
                "axtree": _clip(r.get("axtree_snapshot", ""), 1800),
            })

    # -- sft ------------------------------------------------------------------#
    sft_rows = _jsonl(root / "train" / "sft" / "revact_sft.jsonl")
    test_ids = {r.get("sample_id") for r in _jsonl(root / "train" / "splits" / "sft_test.jsonl")}
    sft = []
    for r in sft_rows:
        goal, obs = _goal_obs(r["messages"][1]["content"])
        m = r.get("meta", {})
        sft.append({
            "id": r.get("sample_id", ""), "action_type": m.get("action_type", ""),
            "variant": m.get("variant", ""), "style": m.get("constraint_style", ""),
            "decision": m.get("decision", ""), "reversibility": m.get("reversibility", ""),
            "split": "test" if r.get("sample_id") in test_ids else "train",
            "goal": goal, "obs": _clip(obs, 2400),
            "assistant": r["messages"][2]["content"],
        })

    # -- dpo ------------------------------------------------------------------#
    dpo = []
    for r in _jsonl(root / "train" / "dpo" / "revact_dpo.jsonl"):
        goal, _ = _goal_obs(r["prompt"][1]["content"]) if r.get("prompt") else ("", "")
        m = r.get("meta", {})
        dpo.append({
            "id": r.get("pair_id", ""), "pair_type": m.get("pair_type", ""),
            "action_type": m.get("action_type", ""), "variant": m.get("variant", ""),
            "goal": goal, "chosen": r.get("chosen", ""), "rejected": r.get("rejected", ""),
        })

    # -- overview -------------------------------------------------------------#
    dec_matrix = collections.Counter((s["action_type"], s["variant"], s["decision"])
                                     for s in sft)
    overview = {
        "n_sft": len(sft), "n_dpo": len(dpo),
        "n_grounded_runs": len(grounded_rows), "n_manifest": len(manifest),
        "n_classes": len(latest), "n_states": len(states),
        "n_traj": len(trajectories),
        "n_traj_success": sum(t["success"] for t in trajectories),
        "n_shots": max_shots - shots_budget["n"],
        "labels": dict(collections.Counter(v.get("label") for v in latest.values())),
        "pair_types": dict(collections.Counter(p["pair_type"] for p in dpo)),
        "styles": dict(collections.Counter(s["style"] for s in sft)),
        "splits": dict(collections.Counter(s["split"] for s in sft)),
        "decision_matrix": [{"action_type": a, "variant": v, "decision": d, "n": n}
                            for (a, v, d), n in sorted(dec_matrix.items())],
        "effective_labels": {at: r.get("label") for at, r in latest.items()},
    }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "site": config.SITE, "data_root": str(root),
        "overview": overview, "grounded": grounded, "trajectories": trajectories,
        "states": states, "sft": sft, "dpo": dpo,
    }


# --------------------------------------------------------------------------- #
# page template
# --------------------------------------------------------------------------- #
_CSS = r"""
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#f6f7f9;--card:#fff;--card2:#fafbfd;--line:#e2e5ec;--line2:#cdd2dc;
  --ink:#1f2937;--dim:#5f6b7c;--mut:#94a0b1;
  --acc:#3f3fb4;--acc-bg:rgba(63,63,180,.05);
  --rev:#0e7a5f;--revc:#0e7490;--part:#b45309;--irr:#b3261e;--unk:#64748b;
  --exe:#0e7a5f;--ver:#1d4ed8;--con:#b45309;--avo:#b3261e;
  --r:8px;--mono:'SF Mono','JetBrains Mono','Cascadia Code',Consolas,monospace;
}
html{scrollbar-width:thin;scrollbar-color:var(--line2) transparent}
body{background:var(--bg);color:var(--ink);min-height:100vh;
  font:15px/1.55 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',sans-serif}
::-webkit-scrollbar{width:6px;height:6px}::-webkit-scrollbar-thumb{background:var(--line2);border-radius:3px}
.wrap{max-width:1280px;margin:0 auto;padding:18px 20px 60px}
header{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;margin-bottom:4px}
header h1{font-size:1.22rem;font-weight:700;color:#32329b;letter-spacing:-.01em}
header h1 span{color:var(--dim);font-weight:400;font-size:.95rem}
header .meta{color:var(--mut);font-size:.72rem;font-variant-numeric:tabular-nums}
.pipe{display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin:10px 0 14px;font-size:.7rem;color:var(--dim)}
.pipe .st{background:var(--card);border:1px solid var(--line);border-radius:999px;padding:2px 10px;white-space:nowrap}
.pipe .st b{font-variant-numeric:tabular-nums;color:var(--ink)}
.pipe .ar{color:var(--mut)}
nav{display:flex;gap:0;border-bottom:1px solid var(--line);margin-bottom:16px;flex-wrap:wrap}
nav button{padding:8px 16px 10px;border:none;background:none;color:var(--mut);cursor:pointer;
  font:inherit;font-size:.8rem;font-weight:500;position:relative}
nav button:hover{color:var(--ink)}
nav button.on{color:var(--acc);font-weight:600}
nav button.on::after{content:'';position:absolute;left:8px;right:8px;bottom:-1px;height:2px;background:var(--acc);border-radius:1px}
nav button:focus-visible,.list button:focus-visible,.fl select:focus-visible{outline:2px solid var(--acc);outline-offset:2px}
section{display:none}section.on{display:block}
.tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px;margin-bottom:18px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:var(--r);padding:12px 14px}
.tile b{display:block;font-size:1.5rem;font-weight:700;font-variant-numeric:tabular-nums;letter-spacing:-.02em}
.tile span{font-size:.68rem;color:var(--dim);text-transform:uppercase;letter-spacing:.06em}
.panes{display:grid;grid-template-columns:290px 1fr;gap:14px;align-items:start}
@media(max-width:820px){.panes{grid-template-columns:1fr}}
.list{background:var(--card);border:1px solid var(--line);border-radius:var(--r);overflow:hidden;max-height:76vh;overflow-y:auto}
.list button{display:block;width:100%;text-align:left;padding:9px 12px;border:none;border-bottom:1px solid var(--line);
  background:none;cursor:pointer;font:inherit;font-size:.78rem;color:var(--ink)}
.list button:last-child{border-bottom:none}
.list button:hover{background:var(--card2)}
.list button.on{background:var(--acc-bg);box-shadow:inset 2px 0 0 var(--acc)}
.list .sub{display:block;color:var(--mut);font-size:.68rem;margin-top:1px;font-family:var(--mono)}
.detail{background:var(--card);border:1px solid var(--line);border-radius:var(--r);padding:16px 18px;min-height:200px}
.detail h3{font-size:.95rem;margin-bottom:10px;display:flex;align-items:center;gap:8px;flex-wrap:wrap;text-wrap:balance}
.badge{display:inline-block;padding:1px 9px;border-radius:999px;font-size:.66rem;font-weight:600;
  letter-spacing:.04em;border:1px solid transparent;vertical-align:1px}
.b-REVERSIBLE{color:var(--rev);background:rgba(14,122,95,.07);border-color:rgba(14,122,95,.25)}
.b-REVERSIBLE_WITH_COST{color:var(--revc);background:rgba(14,116,144,.07);border-color:rgba(14,116,144,.25)}
.b-PARTIALLY_RECOVERABLE{color:var(--part);background:rgba(180,83,9,.07);border-color:rgba(180,83,9,.25)}
.b-IRREVERSIBLE{color:var(--irr);background:rgba(179,38,30,.07);border-color:rgba(179,38,30,.3)}
.b-UNKNOWN,.b-NO_EFFECT{color:var(--unk);background:rgba(100,116,139,.08);border-color:rgba(100,116,139,.25)}
.b-EXECUTE{color:var(--exe);background:rgba(14,122,95,.07);border-color:rgba(14,122,95,.25)}
.b-VERIFY{color:var(--ver);background:rgba(29,78,216,.06);border-color:rgba(29,78,216,.25)}
.b-CONFIRM{color:var(--con);background:rgba(180,83,9,.07);border-color:rgba(180,83,9,.25)}
.b-AVOID{color:var(--avo);background:rgba(179,38,30,.07);border-color:rgba(179,38,30,.3)}
.b-plain{color:var(--dim);background:var(--card2);border-color:var(--line)}
.b-test{color:#7c3aed;background:rgba(124,58,237,.07);border-color:rgba(124,58,237,.3)}
.kv{display:grid;grid-template-columns:max-content 1fr;gap:3px 14px;font-size:.76rem;margin:8px 0}
.kv dt{color:var(--dim)}.kv dd{font-family:var(--mono);font-size:.72rem;overflow-wrap:anywhere}
pre{background:var(--card2);border:1px solid var(--line);border-radius:6px;padding:10px 12px;
  font-family:var(--mono);font-size:.7rem;line-height:1.5;overflow-x:auto;white-space:pre-wrap;overflow-wrap:anywhere}
.seq{font-family:var(--mono);font-size:.73rem;line-height:1.65;background:var(--card2);
  border:1px solid var(--line);border-radius:6px;padding:10px 12px;overflow-x:auto}
.seq .t{color:var(--mut)}
.seq .f-observation{color:#475569}.seq .f-reasoning{color:#334155}
.seq .f-prediction{color:var(--revc)}
.seq .f-reversibility{font-weight:700}.seq .f-decision{font-weight:700}
.seq .f-answer{color:var(--acc);font-weight:600}
.bars{margin:6px 0 14px}
.bar{display:grid;grid-template-columns:190px 1fr 46px;gap:10px;align-items:center;font-size:.74rem;margin:3px 0}
.bar .tr{background:var(--card2);border:1px solid var(--line);border-radius:4px;height:12px;overflow:hidden}
.bar .fi{height:100%;border-radius:3px;background:var(--acc);opacity:.75}
.bar .n{text-align:right;font-variant-numeric:tabular-nums;color:var(--dim)}
table{border-collapse:collapse;font-size:.76rem;width:100%}
th{color:var(--dim);font-weight:600;text-transform:uppercase;font-size:.62rem;letter-spacing:.06em;text-align:left}
th,td{padding:5px 10px;border-bottom:1px solid var(--line)}
td{font-variant-numeric:tabular-nums}
.tbl{background:var(--card);border:1px solid var(--line);border-radius:var(--r);padding:6px 8px;overflow-x:auto;margin-bottom:16px}
h2{font-size:.8rem;color:var(--dim);text-transform:uppercase;letter-spacing:.07em;margin:16px 0 8px}
.fl{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px;align-items:center}
.fl select{font:inherit;font-size:.76rem;padding:4px 8px;border:1px solid var(--line);border-radius:6px;
  background:var(--card);color:var(--ink)}
.fl label{font-size:.68rem;color:var(--dim);text-transform:uppercase;letter-spacing:.05em}
.shots{display:flex;gap:8px;overflow-x:auto;padding:6px 0}
.shots figure{margin:0;flex:0 0 auto}
.shots img{max-height:190px;border:1px solid var(--line);border-radius:6px;display:block}
.shots figcaption{font-size:.64rem;color:var(--mut);font-family:var(--mono);text-align:center;margin-top:3px}
.undo{margin:8px 0 8px 4px;padding-left:14px;border-left:2px solid rgba(14,122,95,.35)}
.undo li{font-family:var(--mono);font-size:.72rem;margin:3px 0;color:var(--ink);list-style:none;position:relative}
.undo li::before{content:'↺';position:absolute;left:-15px;color:var(--rev);font-size:.7rem}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}
@media(max-width:900px){.pair{grid-template-columns:1fr}}
.pair .col h4{font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px}
.pair .ok h4{color:var(--rev)}.pair .bad h4{color:var(--irr)}
.pair .ok .seq{border-color:rgba(14,122,95,.3)}.pair .bad .seq{border-color:rgba(179,38,30,.3)}
details{margin:8px 0}
details summary{cursor:pointer;font-size:.74rem;color:var(--dim)}
.goal{background:var(--acc-bg);border:1px solid rgba(63,63,180,.15);border-radius:6px;
  padding:8px 12px;font-size:.8rem;margin:8px 0}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:6px;vertical-align:1px}
.dot.ok{background:var(--rev)}.dot.no{background:var(--mut)}
.steps td:nth-child(2){font-family:var(--mono);font-size:.7rem}
.steps tr{cursor:pointer}.steps tr:hover td{background:var(--card2)}
.steps tr.on td{background:var(--acc-bg)}
.empty{color:var(--mut);font-size:.8rem;padding:24px;text-align:center}
@media(prefers-reduced-motion:no-preference){nav button,.list button{transition:color .12s,background .12s}}
"""

_JS = r"""
const D=JSON.parse(document.getElementById('viz-data').textContent);
const $=(s,r)=>(r||document).querySelector(s), $$=(s,r)=>Array.from((r||document).querySelectorAll(s));
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const badge=(v,cls)=>`<span class="badge b-${cls||esc(v)}">${esc(v)}</span>`;

/* ---- tabs ---- */
$$('nav button').forEach(b=>b.addEventListener('click',()=>{
  $$('nav button').forEach(x=>x.classList.toggle('on',x===b));
  $$('section').forEach(s=>s.classList.toggle('on',s.id==='tab-'+b.dataset.t));
}));

/* ---- assistant sequence renderer ---- */
function seq(text){
  return '<div class="seq">'+esc(text).split('\n').map(ln=>{
    const m=ln.match(/^&lt;(observation|reasoning|prediction|rev_check|reversibility|undo|decision|answer)&gt;(.*)$/);
    if(!m) return `<div class="t">${ln}</div>`;
    let body=m[2];
    if(m[1]==='reversibility'||m[1]==='decision'){
      body=body.replace(/(REVERSIBLE_WITH_COST\(?\d*\)?|PARTIALLY_RECOVERABLE|IRREVERSIBLE|REVERSIBLE|UNKNOWN|NO_EFFECT|EXECUTE|VERIFY|CONFIRM|AVOID)/,
        w=>badge(w,w.replace(/\(.*/,'')));
    }
    return `<div class="f-${m[1]}"><span class="t">&lt;${m[1]}&gt;</span>${body}</div>`;
  }).join('')+'</div>';
}
function bars(el,counts,color){
  const mx=Math.max(1,...Object.values(counts));
  el.innerHTML=Object.entries(counts).sort((a,b)=>b[1]-a[1]).map(([k,v])=>
    `<div class="bar"><span>${badge(k,color?color(k):k)}</span>
     <span class="tr"><span class="fi" style="width:${100*v/mx}%"></span></span>
     <span class="n">${v}</span></div>`).join('');
}
function master(listEl,detailEl,items,label,render){
  if(!items.length){listEl.innerHTML='';detailEl.innerHTML='<div class="empty">该阶段暂无数据</div>';return;}
  listEl.innerHTML=items.map((it,i)=>`<button data-i="${i}">${label(it)}</button>`).join('');
  const sel=i=>{ $$('button',listEl).forEach((b,j)=>b.classList.toggle('on',j===i));
                 detailEl.innerHTML=render(items[i]); };
  $$('button',listEl).forEach(b=>b.addEventListener('click',()=>sel(+b.dataset.i)));
  sel(0);
}

/* ---- overview ---- */
(function(){
  const o=D.overview;
  $('#tiles').innerHTML=[['SFT 样本',o.n_sft],['DPO 偏好对',o.n_dpo],
    ['grounded 动作类',o.n_classes],['探针运行',o.n_grounded_runs],
    ['风险状态',o.n_states],['轨迹',`${o.n_traj_success}/${o.n_traj}`],
    ['内嵌截图',o.n_shots]].map(([k,v])=>`<div class="tile"><b>${v}</b><span>${k}</span></div>`).join('');
  bars($('#lab-bars'),o.labels,k=>k);
  bars($('#pair-bars'),o.pair_types,()=> 'plain');
  bars($('#style-bars'),o.styles,()=> 'plain');
  const rows=o.decision_matrix.map(r=>{
    const rev=o.effective_labels[r.action_type]||'?';
    return `<tr><td>${esc(r.action_type)}</td><td>${badge(rev)}</td>
      <td>${esc(r.variant)}</td><td>${badge(r.decision)}</td><td>${r.n}</td></tr>`;}).join('');
  $('#matrix').innerHTML=`<table><thead><tr><th>action</th><th>grounded 可逆性</th>
    <th>目标变体</th><th>oracle 决策</th><th>n</th></tr></thead><tbody>${rows}</tbody></table>`;
})();

/* ---- grounded ---- */
master($('#g-list'),$('#g-detail'),D.grounded.slice().reverse(),
  g=>`${g.effective?'●':'○'} ${esc(g.action_type)} ${badge(g.label)}
      <span class="sub">${esc(g.timestamp||'legacy row')}</span>`,
  g=>{
    const kv=Object.entries(g.evidence).filter(([k])=>k!=='undo_actions')
      .map(([k,v])=>`<dt>${esc(k)}</dt><dd>${esc(JSON.stringify(v))}</dd>`).join('');
    const undo=(g.undo_actions||[]).map(a=>`<li>${esc(a)}</li>`).join('');
    const shots=(g.shots||[]).map(s=>`<figure><img src="${s.b64}" alt="${esc(s.name)}">
      <figcaption>${esc(s.name)}</figcaption></figure>`).join('');
    return `<h3>${esc(g.action_type)} ${badge(g.label)}
        ${g.commit_mode?badge('commit','IRREVERSIBLE'):badge('dry-run / non-destructive','plain')}
        ${g.effective?badge('当前生效标签','EXECUTE'):''}</h3>
      <dl class="kv"><dt>probe</dt><dd>${esc(g.probe_name||'(legacy)')}</dd>
        <dt>backend 信号</dt><dd>${esc(g.grounding)}</dd>
        <dt>probe_id</dt><dd>${esc(g.probe_id||'—')}</dd>
        <dt>time</dt><dd>${esc(g.timestamp||'—')}</dd>${kv}</dl>
      ${undo?`<h2>undo 轨迹（实测 ${g.undo_actions.length} 步）</h2><ul class="undo">${undo}</ul>`:''}
      ${shots?`<h2>screenshots</h2><div class="shots">${shots}</div>`:''}`;
  });

/* ---- trajectories ---- */
master($('#t-list'),$('#t-detail'),D.trajectories,
  t=>`<span class="dot ${t.success?'ok':'no'}"></span>${esc(t.id)}
      <span class="sub">${t.n_steps} steps · reward ${t.max_reward}</span>`,
  t=>{
    const rows=t.steps.map((s,i)=>`<tr data-i="${i}"><td>${s.i}</td><td>${esc(s.action)}</td>
      <td>${s.reward||''}</td></tr>`).join('');
    return `<h3>${esc(t.id)} ${t.success?badge('success','EXECUTE'):badge('no reward','plain')}</h3>
      ${t.final?`<div class="goal">final: ${esc(t.final)}</div>`:''}
      <div class="tbl"><table class="steps"><thead><tr><th>#</th><th>action</th><th>r</th></tr></thead>
      <tbody>${rows}</tbody></table></div><div id="t-step"></div>`;
  });
document.addEventListener('click',e=>{
  const tr=e.target.closest('#t-detail .steps tr[data-i]'); if(!tr)return;
  const li=$('#t-list button.on'); const t=D.trajectories[+li.dataset.i]; const s=t.steps[+tr.dataset.i];
  $$('#t-detail .steps tr').forEach(x=>x.classList.toggle('on',x===tr));
  $('#t-step').innerHTML=`<dl class="kv"><dt>url</dt><dd>${esc(s.url)}</dd></dl>
    ${s.shot?`<div class="shots"><figure><img src="${s.shot}" alt="step"></figure></div>`:''}
    <details open><summary>axtree（截断）</summary><pre>${esc(s.axtree)}</pre></details>`;
});

/* ---- states ---- */
master($('#s-list'),$('#s-detail'),D.states,
  s=>`${esc(s.name)} <span class="sub">${esc(s.action_type)} · ${esc(s.source)}</span>`,
  s=>`<h3>${esc(s.name)} ${badge(s.action_type,'plain')} ${badge(s.source,'plain')}</h3>
    <dl class="kv"><dt>url</dt><dd>${esc(s.url)}</dd>
      <dt>risky control</dt><dd>${esc(s.risky)}</dd>
      <dt>safe answer</dt><dd>${esc(s.safe_answer)}</dd></dl>
    <details open><summary>axtree snapshot（截断）</summary><pre>${esc(s.axtree)}</pre></details>`);

/* ---- sft (filterable) ---- */
function opts(sel,vals){sel.innerHTML='<option value="">全部</option>'+
  [...new Set(vals)].sort().map(v=>`<option>${esc(v)}</option>`).join('');}
opts($('#f-at'),D.sft.map(s=>s.action_type)); opts($('#f-va'),D.sft.map(s=>s.variant));
opts($('#f-st'),D.sft.map(s=>s.style)); opts($('#f-sp'),D.sft.map(s=>s.split));
function sftApply(){
  const f=id=>$(id).value;
  const items=D.sft.filter(s=>(!f('#f-at')||s.action_type===f('#f-at'))&&
    (!f('#f-va')||s.variant===f('#f-va'))&&(!f('#f-st')||s.style===f('#f-st'))&&
    (!f('#f-sp')||s.split===f('#f-sp')));
  $('#f-n').textContent=items.length+' / '+D.sft.length;
  master($('#q-list'),$('#q-detail'),items,
    s=>`${esc(s.id)} <span class="sub">${badge(s.decision)} ${badge(s.reversibility)}
        ${s.split==='test'?badge('test','test'):''}</span>`,
    s=>`<h3>${esc(s.id)} ${badge(s.decision)} ${badge(s.reversibility)}
        ${badge(s.style,'plain')} ${s.split==='test'?badge('held-out test','test'):badge('train','plain')}</h3>
      <div class="goal">${esc(s.goal)}</div>
      <details><summary>observation（输入 axtree，截断）</summary><pre>${esc(s.obs)}</pre></details>
      <h2>assistant 目标序列</h2>${seq(s.assistant)}`);
}
['#f-at','#f-va','#f-st','#f-sp'].forEach(id=>$(id).addEventListener('change',sftApply));
sftApply();

/* ---- dpo ---- */
opts($('#p-ty'),D.dpo.map(p=>p.pair_type));
function dpoApply(){
  const t=$('#p-ty').value;
  const items=D.dpo.filter(p=>!t||p.pair_type===t);
  $('#p-n').textContent=items.length+' / '+D.dpo.length;
  master($('#p-list'),$('#p-detail'),items,
    p=>`${esc(p.id.replace(/__/g,' · '))} <span class="sub">${esc(p.pair_type)}</span>`,
    p=>`<h3>${badge(p.pair_type,'plain')} ${esc(p.action_type)} / ${esc(p.variant)}</h3>
      <div class="goal">${esc(p.goal)}</div>
      <div class="pair"><div class="col ok"><h4>chosen（grounded-safe）</h4>${seq(p.chosen)}</div>
      <div class="col bad"><h4>rejected（${esc(p.pair_type)}）</h4>${seq(p.rejected)}</div></div>`);
}
$('#p-ty').addEventListener('change',dpoApply); dpoApply();
"""

_BODY = """
<div class="wrap">
<header>
  <h1>IRIS Dataset Explorer <span>— grounded reversibility pipeline</span></h1>
  <span class="meta">site {site} · generated {generated_at} · {n_sft} SFT / {n_dpo} DPO
    · 静态报告（交互式构建工作台：<code>python -m revact.cli serve</code>）</span>
</header>
<div class="pipe">
  <span class="st">S2 采集 <b>{n_traj}</b> 轨迹</span><span class="ar">→</span>
  <span class="st">S3/scale <b>{n_states}</b> 风险状态</span><span class="ar">→</span>
  <span class="st">S5 探针 <b>{n_runs}</b> 次 / <b>{n_classes}</b> 类 grounded</span><span class="ar">→</span>
  <span class="st">S6–S8 <b>{n_sft}</b> SFT + <b>{n_dpo}</b> DPO</span>
</div>
<nav>
  <button class="on" data-t="ov">Overview</button>
  <button data-t="g">Grounded 可逆性</button>
  <button data-t="t">Trajectories</button>
  <button data-t="s">States</button>
  <button data-t="q">SFT 样本</button>
  <button data-t="p">DPO 对</button>
</nav>

<section id="tab-ov" class="on">
  <div class="tiles" id="tiles"></div>
  <h2>决策矩阵（可逆性 × 目标变体 → oracle 决策）</h2>
  <div class="tbl" id="matrix"></div>
  <h2>grounded 可逆性标签分布（当前生效）</h2><div class="bars" id="lab-bars"></div>
  <h2>DPO pair 类型</h2><div class="bars" id="pair-bars"></div>
  <h2>目标措辞风格</h2><div class="bars" id="style-bars"></div>
</section>

<section id="tab-g"><div class="panes">
  <div class="list" id="g-list"></div><div class="detail" id="g-detail"></div>
</div></section>

<section id="tab-t"><div class="panes">
  <div class="list" id="t-list"></div><div class="detail" id="t-detail"></div>
</div></section>

<section id="tab-s"><div class="panes">
  <div class="list" id="s-list"></div><div class="detail" id="s-detail"></div>
</div></section>

<section id="tab-q">
  <div class="fl">
    <label>action <select id="f-at"></select></label>
    <label>variant <select id="f-va"></select></label>
    <label>style <select id="f-st"></select></label>
    <label>split <select id="f-sp"></select></label>
    <span class="meta" id="f-n"></span>
  </div>
  <div class="panes"><div class="list" id="q-list"></div><div class="detail" id="q-detail"></div></div>
</section>

<section id="tab-p">
  <div class="fl"><label>pair type <select id="p-ty"></select></label><span class="meta" id="p-n"></span></div>
  <div class="panes"><div class="list" id="p-list"></div><div class="detail" id="p-detail"></div></div>
</section>
</div>
"""


def render_html(payload: dict, full_document: bool = True) -> str:
    o = payload["overview"]
    body = _BODY.format(
        site=payload["site"], generated_at=payload["generated_at"],
        n_sft=o["n_sft"], n_dpo=o["n_dpo"], n_traj=o["n_traj"],
        n_states=o["n_states"], n_runs=o["n_grounded_runs"], n_classes=o["n_classes"],
    )
    data_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    title = "<title>IRIS Dataset Explorer</title>"
    style = f"<style>{_CSS}</style>"
    scripts = (f'<script type="application/json" id="viz-data">{data_json}</script>\n'
               f"<script>{_JS}</script>")
    if not full_document:
        # Artifact publishing wraps content in its own document skeleton.
        return f"{title}\n{style}\n{body}\n{scripts}"
    return ("<!DOCTYPE html>\n<html lang=\"zh-CN\">\n<head>\n"
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width,initial-scale=1.0">\n'
            f"{title}\n{style}\n</head>\n<body>\n{body}\n{scripts}\n</body>\n</html>")


def build_viz(out_path: Path | None = None, data_root: Path | None = None,
              max_shots: int = 60, full_document: bool = True) -> Path:
    payload = collect_payload(data_root=data_root, max_shots=max_shots)
    out = out_path or (config.OUTPUTS_DIR / "dataset_viz.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(payload, full_document=full_document), encoding="utf-8")
    return out
