/* DOM helpers + shared renderers (badge / seq / bars / master-detail / toast). */
const $ = (s, r) => (r || document).querySelector(s);
const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));
const esc = s => String(s ?? '').replace(/[&<>"]/g,
  c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

const badge = (v, cls) => v === undefined || v === null || v === ''
  ? '' : `<span class="badge b-${cls || esc(v)}">${esc(v)}</span>`;

const stDot = st => `<span class="st-dot st-${esc(st)}"></span>`;
const STATUS_ZH = { not_started: '未开始', running: '运行中', success: '成功',
  failed: '失败', partial: '部分完成', interrupted: '中断', unknown: '未知' };

function toast(msg, kind) {
  const el = document.createElement('div');
  el.textContent = msg;
  if (kind) el.className = kind;
  $('#toast').appendChild(el);
  setTimeout(() => el.remove(), kind === 'err' ? 7000 : 3500);
}

/* Render the tagged assistant sequence with label badges (from dataset_viz). */
function seq(text) {
  return '<div class="seq">' + esc(text).split('\n').map(ln => {
    const m = ln.match(/^&lt;(observation|reasoning|prediction|rev_check|reversibility|undo|decision|answer)&gt;(.*)$/);
    if (!m) return `<div class="t">${ln}</div>`;
    let body = m[2];
    if (m[1] === 'reversibility' || m[1] === 'decision') {
      body = body.replace(/(REVERSIBLE_WITH_COST\(?\d*\)?|PARTIALLY_RECOVERABLE|IRREVERSIBLE|REVERSIBLE|UNKNOWN|NO_EFFECT|EXECUTE|VERIFY|CONFIRM|AVOID)/,
        w => badge(w, w.replace(/\(.*/, '')));
    }
    return `<div class="f-${m[1]}"><span class="t">&lt;${m[1]}&gt;</span>${body}</div>`;
  }).join('') + '</div>';
}

function bars(counts, colorFn) {
  const entries = Object.entries(counts || {});
  if (!entries.length) return '<div class="empty">无数据</div>';
  const mx = Math.max(1, ...entries.map(e => e[1]));
  return '<div class="bars">' + entries.sort((a, b) => b[1] - a[1]).map(([k, v]) =>
    `<div class="bar"><span>${badge(k, colorFn ? colorFn(k) : k)}</span>
     <span class="tr"><span class="fi" style="width:${100 * v / mx}%"></span></span>
     <span class="n">${v}</span></div>`).join('') + '</div>';
}

/* Master-detail list. render(item) -> html; onSelect optional extra hook. */
function master(listEl, detailEl, items, label, render, onSelect) {
  if (!items.length) {
    listEl.innerHTML = '';
    detailEl.innerHTML = '<div class="empty">该阶段暂无数据</div>';
    return;
  }
  listEl.innerHTML = items.map((it, i) => `<button data-i="${i}">${label(it)}</button>`).join('');
  const sel = i => {
    $$('button', listEl).forEach((b, j) => b.classList.toggle('on', j === i));
    detailEl.innerHTML = render(items[i], i);
    if (onSelect) onSelect(items[i], i);
  };
  $$('button', listEl).forEach(b => b.addEventListener('click', () => sel(+b.dataset.i)));
  sel(0);
}

function selOpts(vals, current) {
  return ['<option value="">全部</option>']
    .concat([...new Set(vals)].sort().map(v =>
      `<option${v === current ? ' selected' : ''}>${esc(v)}</option>`)).join('');
}

function kvTable(obj) {
  const rows = Object.entries(obj || {}).map(([k, v]) =>
    `<dt>${esc(k)}</dt><dd>${esc(typeof v === 'string' ? v : JSON.stringify(v))}</dd>`);
  return rows.length ? `<dl class="kv">${rows.join('')}</dl>` : '';
}

function fmtTime(iso) { return iso ? String(iso).replace('T', ' ').replace(/\+.*/, '') : '—'; }

function annBadge(ann) {
  if (!ann) return '';
  const st = ann.review_status;
  return (st ? badge(st, st) : badge('已编辑', 'plain')) +
    (ann.note ? ` <span class="mini-note">${esc(ann.note)}</span>` : '');
}

/* Simple review-buttons block reused by several tabs.
   extraFields: html injected before buttons; collect(payloadBase) reads them. */
function reviewBox(kind, targetId, ann, opts) {
  const o = opts || {};
  return `<div class="ann-box" data-kind="${esc(kind)}" data-target="${esc(targetId)}">
    <h4>人工覆核 ${ann ? annBadge(ann) : badge('未覆核', 'plain')}</h4>
    ${o.extra || ''}
    <div class="fl">
      <input type="text" class="ann-note" placeholder="备注（可选）" value="${esc(ann && ann.note || '')}" style="min-width:220px">
      <button class="btn sm ann-btn" data-st="confirmed">确认</button>
      <button class="btn sm ann-btn" data-st="accepted">接受</button>
      <button class="btn sm ann-btn" data-st="needs-review">待复核</button>
      <button class="btn sm danger ann-btn" data-st="rejected">驳回</button>
    </div></div>`;
}

/* Delegated handler: any .ann-btn inside a .ann-box posts an annotation.
   Boxes may add inputs with class ann-field + data-field for extra payload. */
document.addEventListener('click', async e => {
  const btn = e.target.closest('.ann-btn');
  if (!btn) return;
  const box = btn.closest('.ann-box');
  const payload = { review_status: btn.dataset.st };
  const note = $('.ann-note', box);
  if (note && note.value.trim()) payload.note = note.value.trim();
  $$('.ann-field', box).forEach(f => {
    const v = f.type === 'checkbox' ? f.checked : f.value;
    if (v !== '' && v !== undefined) payload[f.dataset.field] = f.type === 'number' ? +v : v;
  });
  const r = await API.annotate(box.dataset.kind, box.dataset.target, payload);
  toast(r.ok ? `已记录标注：${box.dataset.target} → ${payload.review_status}` : `标注失败：${r.error}`,
    r.ok ? 'ok' : 'err');
  if (r.ok && window.APP) APP.refreshCurrent();
});
