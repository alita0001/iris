/* All tab views. Each entry: async render(container). Data always comes from
   the API (pipeline artifacts + annotation overlays) — nothing is baked in. */

const LABELS = ['REVERSIBLE', 'REVERSIBLE_WITH_COST', 'PARTIALLY_RECOVERABLE',
  'IRREVERSIBLE', 'NO_EFFECT', 'UNKNOWN'];
const KS_TYPES = ['bottleneck', 'precondition', 'irreversible-risk',
  'goal-progress', 'constraint-sensitive'];
const CONSTRAINT_TYPES = ['safety', 'resource', 'instruction', 'environment',
  'temporal', 'reversibility'];

/* ------------------------------------------------------------ pipeline -- */
const CHAINS = {
  mock: [['env_init', 'check'], ['collect', 'collect_mock'], ['probe', 'probe_mock'],
    ['qc', 'assemble'], ['qc', 'assemble_multiturn'], ['qc', 'split'],
    ['qc', 'compute'], ['export', 'export']],
  live: [['env_init', 'check'], ['collect', 'collect_live'], ['key_states', 'reach'],
    ['probe', 'probe_live'], ['qc', 'assemble'], ['qc', 'assemble_multiturn'],
    ['qc', 'split'], ['distill', 'distill'], ['qc', 'compute'], ['export', 'export']],
};

const PipelineView = {
  stages: [], sel: null, chain: { running: false, lines: [] },

  async render(el) {
    const r = await API.get('/api/pipeline');
    if (!r.ok) { el.innerHTML = `<div class="empty">${esc(r.error)}</div>`; return; }
    this.stages = r.stages;
    if (!this.sel) this.sel = this.stages[0].id;
    el.innerHTML = `
      <div class="fl">
        <label>全流程链</label>
        <select id="pl-mode"><option value="mock">mock（离线可跑）</option>
          <option value="live">live（需 WebArena + key）</option></select>
        <button class="btn primary" id="pl-run-all">全流程运行</button>
        <button class="btn" id="pl-run-from">从选中阶段继续</button>
        <span class="mini-note" id="pl-chain-note">${this.chain.running ? '链运行中…' : ''}</span>
      </div>
      <div class="flow" id="pl-flow"></div>
      <div class="stage-detail" id="pl-detail"></div>
      ${this.chain.lines.length ? `<h2>链运行日志</h2><pre>${esc(this.chain.lines.join('\n'))}</pre>` : ''}`;
    this.renderFlow(); this.renderDetail();
    $('#pl-run-all').addEventListener('click', () => this.runChain(0));
    $('#pl-run-from').addEventListener('click', () => {
      const mode = $('#pl-mode').value;
      const i = CHAINS[mode].findIndex(([s]) => s === this.sel);
      this.runChain(i < 0 ? 0 : i);
    });
  },

  renderFlow() {
    $('#pl-flow').innerHTML = this.stages.map((s, i) => `
      <div class="stage${s.id === this.sel ? ' on' : ''}" data-id="${s.id}">
        <span class="idx">${i + 1} · ${esc(s.s_ref)}</span>
        ${stDot(s.status)}<span class="badge b-plain">${esc(STATUS_ZH[s.status] || s.status)}</span>
        <span class="impl ${esc(s.implemented)}">${esc(s.implemented)}</span>
        <b>${esc(s.title)}</b>
        <span class="sub">${s.last_job ? esc(s.last_job.action) + ' · ' + esc(STATUS_ZH[s.last_job.status] || '') + ' · ' + fmtTime(s.last_job.finished_at || s.last_job.started_at) : '未运行'}</span>
      </div>`).join('');
    $$('#pl-flow .stage').forEach(d => d.addEventListener('click', () => {
      this.sel = d.dataset.id; this.renderFlow(); this.renderDetail();
    }));
  },

  paramForm(stage, a) {
    const f = [];
    if (a.id === 'collect_mock' || a.id === 'collect_live') f.push('seeds:text:0');
    if (a.id === 'collect_live') f.push('max_steps:number:25', 'wa_judge:select:deepseek|off|openai', 'screenshots:checkbox:1');
    if (a.id === 'probe_live') f.push('screenshots:checkbox:1');
    if (a.id === 'probe_named') f.push('names:text:shopping.add_to_cart');
    if (a.id === 'distill') f.push('limit:number:10');
    if (a.id === 'crawl') f.push('cap:number:40');
    if (a.id === 'scale') f.push('n_place_order:number:6');
    return f.map(spec => {
      const [name, type, dflt] = spec.split(':');
      if (type === 'select') {
        const opts = dflt.split('|').map(o => `<option>${o}</option>`).join('');
        return `<label>${name}</label><select class="p-field" data-p="${name}">${opts}</select>`;
      }
      if (type === 'checkbox') return `<label>${name}</label><input type="checkbox" class="p-field" data-p="${name}" checked>`;
      return `<label>${name}</label><input type="${type}" class="p-field" data-p="${name}" value="${dflt}" style="width:110px">`;
    }).join('');
  },

  renderDetail() {
    const s = this.stages.find(x => x.id === this.sel);
    if (!s) return;
    const arts = s.artifacts.map(a => `<tr>
      <td>${esc(a.path)}${a.missing ? ' ' + badge('缺失', 'plain') : ''}</td>
      <td>${a.rows ?? '—'}</td>
      <td>${a.mtime ? new Date(a.mtime * 1000).toISOString().slice(0, 19).replace('T', ' ') : '—'}</td></tr>`).join('');
    const actions = s.actions.map(a => `
      <div class="fl" data-act="${a.id}">
        <button class="btn${a.kind === 'placeholder' ? '' : ' primary'} sm run-btn"
          data-stage="${s.id}" data-action="${a.id}">${esc(a.label)}</button>
        ${a.kind === 'placeholder' ? badge('placeholder', 'needs-review') : ''}
        ${a.needs.map(n => badge(n.replace('key:', 'key: '), 'plain')).join('')}
        ${this.paramForm(s, a)}
        ${a.description ? `<span class="mini-note">${esc(a.description)}</span>` : ''}
      </div>`).join('');
    $('#pl-detail').innerHTML = `
      <h3>${stDot(s.status)}${esc(s.title)} <span class="impl ${esc(s.implemented)}">${esc(s.implemented)}</span>
        <span class="badge b-plain">${esc(s.s_ref)}</span></h3>
      <p class="note">${esc(s.description)}</p>
      <h2>操作</h2>${actions}
      <h2>产物</h2><div class="tbl"><table><thead><tr><th>路径</th><th>行数</th><th>更新时间</th></tr></thead>
        <tbody>${arts}</tbody></table></div>
      ${s.last_job ? `<h2>最近一次运行</h2><div class="fl">
        ${stDot(s.last_job.status)}<span class="mini-note">${esc(s.last_job.job_id)} · ${esc(s.last_job.action)} · ${fmtTime(s.last_job.started_at)}</span>
        <button class="btn sm" onclick="APP.showJob('${esc(s.last_job.job_id)}')">查看日志</button></div>` : ''}
      <div id="pl-result"></div>`;
    $$('#pl-detail .run-btn').forEach(b => b.addEventListener('click', async () => {
      const row = b.closest('[data-act]');
      const params = {};
      $$('.p-field', row).forEach(f => {
        params[f.dataset.p] = f.type === 'checkbox' ? f.checked
          : f.dataset.p === 'names' ? f.value.split(',').map(x => x.trim()).filter(Boolean)
            : f.value;
      });
      b.disabled = true;
      const r = await API.runStage(b.dataset.stage, b.dataset.action, params);
      b.disabled = false;
      if (r.ok && r.job) {
        toast(`已启动 ${r.job.job_id}`, 'ok'); APP.showJob(r.job.job_id);
      } else if (r.ok) {
        toast(r.result && r.result.note || '完成', 'ok');
        $('#pl-result').innerHTML = `<h2>结果</h2><pre>${esc(JSON.stringify(r.result, null, 1).slice(0, 6000))}</pre>`;
        APP.refreshCurrent();
      } else {
        toast(r.error + (r.extension_point ? `（扩展点：${r.extension_point}）` : ''), 'err');
      }
    }));
  },

  async runChain(fromIdx) {
    if (this.chain.running) { toast('链已在运行', 'err'); return; }
    const mode = $('#pl-mode').value;
    const chain = CHAINS[mode].slice(fromIdx);
    this.chain = { running: true, lines: [`[chain:${mode}] ${chain.map(c => c.join('.')).join(' → ')}`] };
    toast(`链启动（${chain.length} 步）`);
    for (const [stage, action] of chain) {
      this.chain.lines.push(`→ ${stage}.${action} …`);
      const r = await API.runStage(stage, action, {});
      if (!r.ok) {
        this.chain.lines.push(`✗ ${stage}.${action}: ${r.error}`);
        toast(`链中断于 ${stage}.${action}: ${r.error}`, 'err');
        break;
      }
      if (r.job) {
        const final = await this.waitJob(r.job.job_id);
        this.chain.lines.push(`  ${final.status === 'success' ? '✓' : '✗'} ${r.job.job_id} (${final.status})`);
        if (final.status !== 'success') { toast(`链中断：${r.job.job_id} ${final.status}`, 'err'); break; }
      } else {
        this.chain.lines.push(`  ✓ ${r.result && r.result.note || 'ok'}`);
      }
    }
    this.chain.running = false;
    this.chain.lines.push('[chain] 结束');
    APP.refreshCurrent();
  },

  async waitJob(jid) {
    for (;;) {
      await new Promise(res => setTimeout(res, 1500));
      const r = await API.get('/api/jobs/' + jid);
      if (!r.ok) return { status: 'failed' };
      if (r.job.status !== 'running') return r.job;
    }
  },
};

/* -------------------------------------------------------------- config -- */
const ConfigView = {
  async render(el) {
    const [cfg, health] = await Promise.all([API.get('/api/config'), API.get('/api/health')]);
    if (!cfg.ok) { el.innerHTML = `<div class="empty">${esc(cfg.error)}</div>`; return; }
    const s = cfg.settings, sum = health.summary || {};
    const model = (role, m) => `
      <div class="fcard" data-role="${role}">
        <h4>${{ policy: '策略模型（collect rollout）', teacher: 'Teacher 模型（S7 蒸馏）', judge: 'Judge / Verifier（WA reward judge）' }[role]}
          ${m.api_key_set ? badge('key 已在内存', 'accepted') : badge('key 未设置', 'plain')}</h4>
        <div class="frow"><label>provider</label>
          <select class="c-f" data-f="provider"><option${m.provider === 'deepseek' ? ' selected' : ''}>deepseek</option>
            <option${m.provider === 'openai' ? ' selected' : ''}>openai</option>
            <option${m.provider === 'custom' ? ' selected' : ''}>custom</option></select></div>
        <div class="frow"><label>base_url</label><input type="text" class="c-f" data-f="base_url" value="${esc(m.base_url || '')}" placeholder="默认 DeepSeek v1"></div>
        <div class="frow"><label>model</label><input type="text" class="c-f" data-f="model" value="${esc(m.model || '')}" placeholder="deepseek-chat / deepseek-v4-pro"></div>
        <div class="frow"><label>api_key_env</label><input type="text" class="c-f" data-f="api_key_env" value="${esc(m.api_key_env || '')}"></div>
        <div class="frow"><label>api_key</label><input type="password" class="c-f" data-f="api_key" placeholder="${m.api_key_set ? '已设置（留空保持）' : '只存内存，不落盘'}"></div>
        ${role !== 'judge' ? `
        <div class="frow"><label>temperature</label><input type="number" step="0.1" min="0" max="2" class="c-f" data-f="temperature" value="${m.temperature ?? 0}"></div>
        <div class="frow"><label>top_p</label><input type="number" step="0.05" min="0" max="1" class="c-f" data-f="top_p" value="${m.top_p ?? 1}"></div>
        <div class="frow"><label>max_tokens</label><input type="number" class="c-f" data-f="max_tokens" value="${m.max_tokens ?? 4096}"></div>` : `
        <div class="frow"><label>judge 模式</label><select class="c-f" data-f="mode">
          ${['deepseek', 'openai', 'off'].map(o => `<option${m.mode === o ? ' selected' : ''}>${o}</option>`).join('')}</select></div>`}
      </div>`;
    el.innerHTML = `
      <div class="tiles">
        <div class="tile"><b>${sum.n_traj ?? '—'}</b><span>轨迹（${sum.n_traj_success ?? 0} 成功）</span></div>
        <div class="tile"><b>${sum.n_grounded_classes ?? '—'}</b><span>grounded 动作类</span></div>
        <div class="tile"><b>${sum.n_sft ?? '—'} / ${sum.n_dpo ?? '—'}</b><span>SFT / DPO</span></div>
        <div class="tile"><b>${health.live_ready ? 'LIVE' : 'OFFLINE'}</b><span>WebArena 环境</span></div>
        <div class="tile"><b>${health.n_running ?? 0}</b><span>运行中任务</span></div>
      </div>
      <dl class="kv"><dt>数据根目录</dt><dd>${esc(sum.data_root || '')}</dd>
        <dt>输出目录</dt><dd>${esc(health.outputs_dir || '')}</dd>
        <dt>静态报告</dt><dd>${esc(health.viz_report || '')}</dd>
        <dt>最近一次任务</dt><dd>${health.last_job ? esc(health.last_job.job_id + ' · ' + health.last_job.status + ' · ' + fmtTime(health.last_job.started_at)) : '—'}</dd>
        <dt>本地配置文件</dt><dd>${esc(cfg.local_config)}${cfg.local_config_exists ? '' : '（尚未保存）'}</dd></dl>
      <p class="warn-note">安全约定：api_key 只存服务进程内存并注入子进程环境变量；「保存到本地」写入 configs/workbench.local.json（已 gitignore）且会剥离 key 值，只保留 env 变量名。</p>
      <div class="form-grid">
        ${model('policy', s.models.policy)}${model('teacher', s.models.teacher)}${model('judge', s.models.judge)}
        <div class="fcard" data-role="run">
          <h4>运行参数</h4>
          <div class="frow"><label>task_file</label><input type="text" class="c-f" data-f="task_file" value="${esc(s.run.task_file || '')}"></div>
          <div class="frow"><label>seeds</label><input type="text" class="c-f" data-f="seeds" value="${esc(s.run.seeds || '0')}"></div>
          <div class="frow"><label>max_steps</label><input type="number" class="c-f" data-f="max_steps" value="${s.run.max_steps ?? 25}"></div>
          <div class="frow"><label>采样数量</label><input type="number" class="c-f" data-f="sample_limit" value="${s.run.sample_limit ?? 10}"></div>
          <div class="frow"><label>截图</label><input type="checkbox" class="c-f" data-f="screenshots" ${s.run.screenshots ? 'checked' : ''}></div>
          <div class="frow"><label>data_root 覆盖</label><input type="text" class="c-f" data-f="data_root" value="${esc(s.run.data_root || '')}" placeholder="留空用默认 data/"></div>
          <div class="frow"><label>输出目录</label><input type="text" class="c-f" data-f="output_dir" value="${esc(s.run.output_dir || '')}"></div>
        </div>
        <div class="fcard" data-role="env">
          <h4>环境（任务环境选择）</h4>
          <div class="frow"><label>WA_SHOPPING</label><input type="text" class="c-f" data-f="WA_SHOPPING" value="${esc(s.env.WA_SHOPPING || '')}" placeholder="留空则用 shell env"></div>
          <div class="frow"><label>WA_SHOPPING_ADMIN</label><input type="text" class="c-f" data-f="WA_SHOPPING_ADMIN" value="${esc(s.env.WA_SHOPPING_ADMIN || '')}"></div>
          <div class="frow"><label>WA_REDDIT</label><input type="text" class="c-f" data-f="WA_REDDIT" value="${esc(s.env.WA_REDDIT || '')}" placeholder="Postmill 镜像，如 http://user2-dind:9999"></div>
          <p class="note">live 采集/探针前请确认已 source scripts/export_webarena_env.sh，或在此填入站点 URL。</p>
        </div>
      </div>
      <div class="fl">
        <button class="btn primary" id="cfg-apply">应用配置（内存）</button>
        <button class="btn" id="cfg-save">保存到本地文件（剥离 key）</button>
        <button class="btn" id="cfg-reload">重新加载</button>
      </div>`;
    const collect = () => {
      const body = { models: {}, run: {}, env: {} };
      $$('[data-role]', el).forEach(card => {
        const role = card.dataset.role;
        const tgt = role === 'run' ? body.run : role === 'env' ? body.env : (body.models[role] = {});
        $$('.c-f', card).forEach(f => {
          let v = f.type === 'checkbox' ? f.checked : f.value;
          if (f.dataset.f === 'api_key' && v === '') return;   // keep existing
          if (f.type === 'number' && v !== '') v = +v;
          tgt[f.dataset.f] = v;
        });
      });
      return body;
    };
    $('#cfg-apply').addEventListener('click', async () => {
      const r = await API.post('/api/config', collect());
      toast(r.ok ? '配置已应用（key 只在内存）' : r.error, r.ok ? 'ok' : 'err');
      if (r.ok) this.render(el);
    });
    $('#cfg-save').addEventListener('click', async () => {
      const r = await API.post('/api/config/save', collect());
      toast(r.ok ? r.note : r.error, r.ok ? 'ok' : 'err');
      if (r.ok) this.render(el);
    });
    $('#cfg-reload').addEventListener('click', () => this.render(el));
  },
};

/* ------------------------------------------------------------- prompts -- */
const PromptsView = {
  async render(el) {
    const r = await API.get('/api/prompts');
    if (!r.ok) { el.innerHTML = `<div class="empty">${esc(r.error)}</div>`; return; }
    const card = (p, i) => {
      const val = p.kind === 'list' ? p.value.join('\n') : p.value;
      const rows = Math.min(18, Math.max(4, val.split('\n').length + (p.kind === 'list' ? 1 : 2)));
      return `
      <div class="fcard" data-pid="${esc(p.id)}">
        <h4>${esc(p.title)} ${p.overridden ? badge('已覆盖', 'accepted') : badge('默认', 'plain')}</h4>
        <p class="note">${esc(p.description)}</p>
        <p class="mini-note">使用位置：${esc(p.used_by)}${p.placeholders.length
          ? ' · 必需占位符：' + p.placeholders.map(esc).join(' ') : ''}${p.kind === 'list'
          ? ' · 每行一条模板' : ''}</p>
        <textarea class="p-text" rows="${rows}" spellcheck="false">${esc(val)}</textarea>
        <div class="fl">
          <button class="btn primary p-save">保存覆盖</button>
          <button class="btn p-reset" ${p.overridden ? '' : 'disabled'}>恢复默认</button>
          <button class="btn p-diff">对比默认</button>
        </div>
        <pre class="p-default" hidden>${esc(p.kind === 'list' ? p.default.join('\n') : p.default)}</pre>
      </div>`;
    };
    el.innerHTML = `
      <p class="warn-note">这里管理 pipeline 全部 LLM prompt 与目标模板池：采集策略模型、teacher 蒸馏、
      训练/部署共用的 agent 系统提示词。覆盖保存在 ${esc(r.overrides_file)}（当前指纹
      <code>${esc(r.fingerprint)}</code>，会写入之后物化样本的 meta.prompts_fp 以便溯源）。
      改动 <b>agent_system / 模板池</b> 后必须重跑 assemble（+ assemble-multiturn + split）
      重物化训练数据；改动采集/蒸馏 prompt 对之后的新 job 立即生效，无需重启。</p>
      <div class="form-grid">${r.items.map(card).join('')}</div>`;
    $$('[data-pid]', el).forEach(cardEl => {
      const pid = cardEl.dataset.pid;
      const item = r.items.find(p => p.id === pid);
      $('.p-save', cardEl).addEventListener('click', async () => {
        const raw = $('.p-text', cardEl).value;
        const value = item.kind === 'list'
          ? raw.split('\n').map(s => s.trim()).filter(Boolean) : raw;
        const res = await API.post('/api/prompts', { id: pid, value });
        toast(res.ok ? `已保存（新指纹 ${res.fingerprint}）· ${res.note || ''}` : res.error,
          res.ok ? 'ok' : 'err');
        if (res.ok) this.render(el);
      });
      $('.p-reset', cardEl).addEventListener('click', async () => {
        const res = await API.post('/api/prompts/reset', { id: pid });
        toast(res.ok ? '已恢复默认' : res.error, res.ok ? 'ok' : 'err');
        if (res.ok) this.render(el);
      });
      $('.p-diff', cardEl).addEventListener('click', () => {
        const pre = $('.p-default', cardEl);
        pre.hidden = !pre.hidden;
      });
    });
  },
};

/* -------------------------------------------------------- trajectories -- */
const TrajView = {
  filter: 'all',
  async render(el) {
    const [r, ann] = await Promise.all([
      API.get('/api/trajectories'), API.get('/api/annotations/trajectory')]);
    if (!r.ok) { el.innerHTML = `<div class="empty">${esc(r.error)}</div>`; return; }
    const anns = ann.ok ? ann.effective : {};
    const items = r.items.filter(t =>
      this.filter === 'all' ? true
        : this.filter === 'success' ? t.success
          : this.filter === 'failed' ? !t.success && !t.anomalous : t.anomalous);
    el.innerHTML = `
      <div class="fl"><label>筛选</label>
        <select id="tj-f">${['all', 'success', 'failed', 'anomalous'].map(o =>
          `<option value="${o}"${o === this.filter ? ' selected' : ''}>${{ all: '全部', success: '成功', failed: '失败', anomalous: '异常(截断)' }[o]}</option>`).join('')}</select>
        <span class="mini-note">${items.length} / ${r.items.length} 条 · 选中 ${Object.values(anns).filter(a => a.selected).length} 条进入下一阶段</span></div>
      <div class="panes"><div class="list" id="tj-list"></div><div class="detail" id="tj-detail"></div></div>`;
    $('#tj-f').addEventListener('change', e => { this.filter = e.target.value; this.render(el); });
    master($('#tj-list'), $('#tj-detail'), items,
      t => `<span class="dot ${t.success ? 'ok' : t.anomalous ? 'bad' : 'no'}"></span>${esc(t.trajectory_id)}
        ${anns[t.trajectory_id] && anns[t.trajectory_id].selected ? badge('已选', 'accepted') : ''}
        <span class="sub">${t.n_steps ?? '?'} steps · reward ${t.max_reward} · seed ${t.seed ?? '?'}</span>`,
      t => `<h3>${esc(t.trajectory_id)} ${t.success ? badge('success', 'EXECUTE') : badge('no reward', 'plain')}</h3>
        <div id="tj-body" class="loading">加载 step trace…</div>`,
      t => this.loadDetail(t, anns));
  },

  async loadDetail(t, anns) {
    const r = await API.get('/api/trajectories/' + encodeURIComponent(t.trajectory_id));
    const body = $('#tj-body');
    if (!body) return;
    body.classList.remove('loading');
    if (!r.ok) { body.innerHTML = `<div class="empty">${esc(r.error)}</div>`; return; }
    const tr = r.trajectory, m = tr.meta || {};
    const sel = anns[t.trajectory_id] && anns[t.trajectory_id].selected;
    const rows = tr.steps.map((s, i) => `<tr data-i="${i}"><td>${s.step_id}</td>
      <td style="font-family:var(--mono);font-size:.7rem">${esc(s.action)}</td>
      <td>${s.reward || ''}</td><td>${s.terminated ? '✓' : ''}</td></tr>`).join('');
    body.innerHTML = `
      <dl class="kv"><dt>task</dt><dd>${esc(m.task_id || '')}</dd>
        <dt>模型响应(final)</dt><dd>${esc((m.final_model_response || '').slice(0, 300))}</dd>
        <dt>key states</dt><dd>${tr.key_states.length} 个（关键状态页可覆核）</dd></dl>
      <div class="fl">
        <button class="btn sm ${sel ? '' : 'primary'}" id="tj-sel">${sel ? '取消选择' : '选入下一阶段（关键状态采集）'}</button>
        <button class="btn sm" onclick="APP.go('keystates')">查看关键状态 →</button></div>
      <div class="tbl"><table class="steps"><thead><tr><th>#</th><th>action</th><th>r</th><th>done</th></tr></thead>
        <tbody>${rows}</tbody></table></div>
      <div id="tj-step" class="note">点击任意 step 查看 observation / 截图</div>`;
    $('#tj-sel').addEventListener('click', async () => {
      const rr = await API.annotate('trajectory', t.trajectory_id, { selected: !sel });
      toast(rr.ok ? '已记录' : rr.error, rr.ok ? 'ok' : 'err');
      if (rr.ok) APP.refreshCurrent();
    });
    $$('#tj-body .steps tr[data-i]').forEach(row => row.addEventListener('click', () => {
      $$('#tj-body .steps tr').forEach(x => x.classList.toggle('on', x === row));
      const s = tr.steps[+row.dataset.i];
      $('#tj-step').innerHTML = `
        <dl class="kv"><dt>url</dt><dd>${esc(s.url_after)}</dd></dl>
        ${s.screenshot ? `<div class="shots"><figure><img src="/api/screenshot?path=${encodeURIComponent(s.screenshot)}" alt="step"><figcaption>${esc(s.screenshot.split('/').pop())}</figcaption></figure></div>` : ''}
        <details open><summary>observation axtree（截断）</summary><pre>${esc(s.axtree)}</pre></details>`;
    }));
  },
};

/* ----------------------------------------------------------- keystates -- */
const KeyStatesView = {
  ftype: '', fstatus: '',
  async render(el) {
    const r = await API.get('/api/keystates');
    if (!r.ok) { el.innerHTML = `<div class="empty">${esc(r.error)}</div>`; return; }
    const anns = r.annotations || {};
    const types = [...new Set(r.items.flatMap(k => k.afforded_action_types))];
    const items = r.items.filter(k => {
      const a = anns[k.state_id] || {};
      if (this.ftype && !k.afforded_action_types.includes(this.ftype)
          && a.state_type !== this.ftype) return false;
      if (this.fstatus && (a.review_status || '') !== this.fstatus) return false;
      return true;
    });
    el.innerHTML = `
      <div class="fl">
        <label>动作类型/标签</label><select id="ks-t">${selOpts(types.concat(KS_TYPES), this.ftype)}</select>
        <label>覆核状态</label><select id="ks-s">${selOpts(['confirmed', 'accepted', 'rejected', 'needs-review'], this.fstatus)}</select>
        <span class="mini-note">${items.length} / ${r.items.length} 个 key state（来源：S2 关键词规则挖掘）</span></div>
      <div class="panes"><div class="list" id="ks-list"></div><div class="detail" id="ks-detail"></div></div>`;
    $('#ks-t').addEventListener('change', e => { this.ftype = e.target.value; this.render(el); });
    $('#ks-s').addEventListener('change', e => { this.fstatus = e.target.value; this.render(el); });
    master($('#ks-list'), $('#ks-detail'), items,
      k => `${esc(k.state_id)} ${annBadge(anns[k.state_id])}
        <span class="sub">${esc(k.afforded_action_types.join(', '))} · step ${k.step_id}</span>`,
      k => {
        const a = anns[k.state_id] || {};
        return `<h3>${esc(k.state_id)} ${k.afforded_action_types.map(t => badge(t, 'plain')).join(' ')}</h3>
        <dl class="kv"><dt>trajectory</dt><dd>${esc(k.trajectory_id)} @ step ${k.step_id}</dd>
          <dt>task goal</dt><dd>${esc(k.goal)}</dd>
          <dt>url</dt><dd>${esc(k.url)}</dd>
          <dt>replay_prefix</dt><dd>${esc(JSON.stringify(k.replay_prefix).slice(0, 300))}</dd></dl>
        <div class="goal">为什么是关键状态：页面 afford 高危可交互动作（关键词规则命中
          ${k.afforded_action_types.map(t => `<b>${esc(t)}</b>`).join('、')}）；grounding 由 S5 探针细化，此处仅浅层识别。</div>
        <details><summary>observation snapshot（截断）</summary><pre>${esc(k.axtree)}</pre></details>
        ${reviewBox('key_state', k.state_id, a, { extra: `
          <div class="fl"><label>状态类型</label>
            <select class="ann-field" data-field="state_type">${['', ...KS_TYPES].map(t =>
              `<option${(a.state_type || '') === t ? ' selected' : ''}>${t}</option>`).join('')}</select>
            <label>置信度</label>
            <input type="number" class="ann-field" data-field="confidence" data-fieldtype="num"
              min="0" max="1" step="0.05" value="${a.confidence ?? ''}" style="width:80px"></div>` })}`;
      });
  },
};

/* --------------------------------------------------------- constraints -- */
const ConstraintsView = {
  async render(el) {
    const [tpl, prev, sft] = await Promise.all([
      API.get('/api/templates'), API.get('/api/constraints/preview'), API.get('/api/sft')]);
    if (!prev.ok) { el.innerHTML = `<div class="empty">${esc(prev.error)}</div>`; return; }
    const anns = tpl.annotations || {};
    const decBy = {};
    (sft.items || []).forEach(s => { decBy[s.sample_id] = s.decision; });
    const pool = (name, rows) => `<details${name === 'explicit' ? ' open' : ''}><summary>${{ explicit: '显式约束模板', implicit: '隐式约束模板（无 do-not token）', request: '请求措辞模板' }[name]}（${rows.length}）</summary>
      <div class="tbl"><table><tbody>${rows.map((t, i) => `<tr><td class="mini-note">${i}</td><td>${esc(t)}</td></tr>`).join('')}</tbody></table></div></details>`;
    el.innerHTML = `
      <p class="note">约束注入 = assemble.build_goal 按 (state, variant) 确定性抽取模板（此页预览即真实注入逻辑）。
      人工编辑保存为 overlay（constraint 标注），导出时体现在审计字段；修改后需重跑 assemble 物化。</p>
      ${pool('explicit', tpl.explicit || [])}${pool('implicit', tpl.implicit || [])}${pool('request', tpl.request || [])}
      <div class="fl">
        <button class="btn primary sm" id="cs-assemble">物化（运行 assemble）</button>
        <span class="mini-note">共 ${prev.previews.length} 个状态可注入（已绑定 grounded 标签）</span></div>
      <div class="panes"><div class="list" id="cs-list"></div><div class="detail" id="cs-detail"></div></div>`;
    $('#cs-assemble').addEventListener('click', async () => {
      const r = await API.runStage('constraints', 'assemble', {});
      if (r.ok && r.job) { toast('assemble 已启动', 'ok'); APP.showJob(r.job.job_id); } else toast(r.error || 'ok', r.ok ? 'ok' : 'err');
    });
    master($('#cs-list'), $('#cs-detail'), prev.previews,
      p => `${esc(p.state)} <span class="sub">${esc(p.action_type)}</span>`,
      p => {
        const variant = (v, g) => {
          const key = `${p.state}__${v}`;
          const a = anns[key] || {};
          const dec = decBy[key];
          return `<div class="fcard">
            <h4>${v === 'constraint' ? '注入约束（violates=true）' : '请求变体（requested=true）'}
              ${badge(g.style, 'plain')} <span class="mini-note">${esc(g.template_id)}</span>
              ${dec ? '→ oracle ' + badge(dec) : ''}</h4>
            <div class="goal">${esc(a.goal_override || g.goal)}${a.goal_override ? ' ' + badge('人工改写', 'needs-review') : ''}</div>
            <div class="ann-box" data-kind="constraint" data-target="${esc(key)}">
              <h4>编辑 ${annBadge(a)}</h4>
              <textarea class="ann-field" data-field="goal_override" placeholder="改写目标文本（留空=用模板）">${esc(a.goal_override || '')}</textarea>
              <div class="fl"><label>约束类型</label>
                <select class="ann-field" data-field="constraint_type">${['', ...CONSTRAINT_TYPES].map(t =>
                  `<option${(a.constraint_type || '') === t ? ' selected' : ''}>${t}</option>`).join('')}</select>
                <input type="text" class="ann-note" placeholder="备注" value="${esc(a.note || '')}" style="min-width:160px">
                <button class="btn sm ann-btn" data-st="accepted">保存</button></div></div></div>`;
        };
        return `<h3>${esc(p.state)} ${badge(p.action_type, 'plain')}</h3>
          <p class="note">预期行为变化：同一状态在两个变体下 oracle 决策不同（见每栏右上角），即注入前后候选动作评价的对照。</p>
          <div class="pair"><div class="col">${variant('constraint', p.variants.constraint)}</div>
          <div class="col">${variant('request', p.variants.request)}</div></div>
          <div class="fl"><button class="btn sm" onclick="APP.go('candidates')">查看该状态候选动作 →</button></div>`;
      });
  },
};

/* ---------------------------------------------------------- candidates -- */
const CandidatesView = {
  state: '',
  async render(el) {
    const states = await API.get('/api/states');
    if (!states.ok) { el.innerHTML = `<div class="empty">${esc(states.error)}</div>`; return; }
    const bound = states.items.filter(s => s.grounded_action_type);
    if (!this.state && bound.length) this.state = bound[0].name;
    el.innerHTML = `
      <div class="fl"><label>状态</label>
        <select id="cd-state">${bound.map(s => `<option${s.name === this.state ? ' selected' : ''}>${esc(s.name)}</option>`).join('')}</select>
        <button class="btn sm" id="cd-regen">重新生成（重算反事实）</button>
        <button class="btn sm" onclick="APP.go('grounded')">送去 undo 标注 →</button>
        <span class="mini-note">共 ${bound.length} 个已绑定 grounded 标签的状态（未绑定的 ${states.items.length - bound.length} 个不产样本）</span></div>
      <div id="cd-body" class="loading">加载中…</div>`;
    $('#cd-state').addEventListener('change', e => { this.state = e.target.value; this.render(el); });
    $('#cd-regen').addEventListener('click', () => this.render(el));
    const r = await API.get('/api/candidates?state=' + encodeURIComponent(this.state));
    const body = $('#cd-body');
    body.classList.remove('loading');
    if (!r.ok || !r.candidates) { body.innerHTML = '<div class="empty">该状态无法生成候选（未绑定 grounded 标签）</div>'; return; }
    const c = r.candidates, anns = r.annotations || {};
    const candRows = c.candidates.map(x => `<tr>
      <td>${badge(x.kind, 'plain')}</td><td>${esc(x.text)}</td>
      <td style="font-family:var(--mono)">${esc(x.raw_action)}</td>
      <td>${x.reversibility ? badge(x.reversibility) : '—'}</td>
      <td>${x.grounded ? badge('grounded', 'accepted') : badge('rule', 'plain')}</td></tr>`).join('');
    const cfRows = c.counterfactuals.map((x, i) => `<tr>
      <td>${badge(x.pair_type, 'plain')}</td><td>${esc(x.variant)}</td>
      <td style="font-family:var(--mono)">${esc(x.raw_action)}</td>
      <td>${badge(x.reversibility_claimed)}</td><td>${badge(x.decision_claimed.split(' ')[0])}</td>
      <td><button class="btn sm cf-show" data-i="${i}">序列</button></td></tr>`).join('');
    const customs = Object.entries(anns).filter(([, v]) => !v.deleted).map(([k, v]) => `<tr>
      <td>${badge('custom', 'needs-review')}</td><td>${esc(v.action_text || '')}</td>
      <td style="font-family:var(--mono)">${esc(v.raw_action || '')}</td>
      <td>${esc(v.note || '')}</td>
      <td><button class="btn sm danger cd-del" data-k="${esc(k)}">删除</button></td></tr>`).join('');
    body.innerHTML = `
      <h2>候选动作（pipeline 真实来源）</h2>
      <div class="tbl"><table><thead><tr><th>类型</th><th>text</th><th>action</th><th>grounded 可逆性</th><th>来源</th></tr></thead>
        <tbody>${candRows}</tbody></table></div>
      <h2>反事实动作（DPO rejected builders，实时计算）</h2>
      <div class="tbl"><table><thead><tr><th>pair type</th><th>variant</th><th>action</th><th>宣称可逆性</th><th>宣称决策</th><th></th></tr></thead>
        <tbody>${cfRows}</tbody></table></div>
      <div id="cd-seq"></div>
      <h2>人工添加候选（overlay，不进自动物化）</h2>
      <div class="tbl"><table><thead><tr><th></th><th>text</th><th>action</th><th>备注</th><th></th></tr></thead>
        <tbody>${customs || '<tr><td colspan=5 class="empty">暂无</td></tr>'}</tbody></table></div>
      <div class="fl">
        <input type="text" id="cd-text" placeholder="动作描述，如 click 'Clear Cart'" style="min-width:220px">
        <input type="text" id="cd-raw" placeholder="raw action，如 click('123')" style="min-width:160px">
        <input type="text" id="cd-note" placeholder="备注/预期后果">
        <button class="btn primary sm" id="cd-add">添加候选</button></div>
      <p class="warn-note">当前 pipeline 无生成式候选提案阶段（S4 placeholder）：以上候选=专家动作+规则安全动作+DPO 反事实；人工添加仅作标注资产，接入自动生成见 docs/workbench.md 扩展点。</p>`;
    $$('.cf-show', body).forEach(b => b.addEventListener('click', () => {
      $('#cd-seq').innerHTML = `<h2>rejected 完整序列</h2>${seq(c.counterfactuals[+b.dataset.i].rendered)}`;
    }));
    $('#cd-add').addEventListener('click', async () => {
      const text = $('#cd-text').value.trim();
      if (!text) { toast('填写动作描述', 'err'); return; }
      const r2 = await API.annotate('candidate', `${this.state}__custom-${Date.now()}`,
        { action_text: text, raw_action: $('#cd-raw').value.trim(), note: $('#cd-note').value.trim() });
      toast(r2.ok ? '已添加' : r2.error, r2.ok ? 'ok' : 'err');
      if (r2.ok) this.render(el);
    });
    $$('.cd-del', body).forEach(b => b.addEventListener('click', async () => {
      const r2 = await API.annotate('candidate', b.dataset.k, { deleted: true });
      if (r2.ok) this.render(el);
    }));
  },
};

/* ------------------------------------------------------------ grounded -- */
const GroundedView = {
  async render(el) {
    const [r, probes] = await Promise.all([API.get('/api/grounded'), API.get('/api/probes')]);
    if (!r.ok) { el.innerHTML = `<div class="empty">${esc(r.error)}</div>`; return; }
    const anns = r.annotations || {};
    const bySite = {};
    (probes.items || []).forEach(p => { (bySite[p.site] = bySite[p.site] || []).push(p); });
    const siteBlocks = Object.keys(bySite).sort().map(site => {
      const rows = bySite[site].map(p => `<tr>
        <td style="font-family:var(--mono)">${esc(p.name)}</td>
        <td>${badge(p.destructive.replace('_', '-'), p.destructive === 'destructive' ? 'IRREVERSIBLE' : 'plain')}</td>
        <td class="mini-note">${esc(p.grounding)}</td>
        <td>${p.live_label ? badge(p.live_label) : badge('未采集', 'plain')}</td>
        <td class="mini-note">${esc(p.expected_spectrum)}</td></tr>`).join('');
      return `<h4 style="margin:.6em 0 .2em">${esc(site)} <span class="mini-note">(${bySite[site].length})</span></h4>
        <div class="tbl"><table><thead><tr><th>probe</th><th>等级</th><th>backend 信号</th><th>当前标签</th><th>预期谱系</th></tr></thead>
          <tbody>${rows}</tbody></table></div>`;
    }).join('');
    el.innerHTML = `
      <div class="fl">
        <button class="btn primary sm" id="gp-mock">mock 探针（离线·全站点）</button>
        <button class="btn sm" id="gp-live">live 非破坏（shopping）</button>
        <button class="btn sm" id="gp-live-reddit">live 非破坏（reddit）</button>
        <span class="warn-note">破坏性探针不在工作台开放：需 CLI 双闸门（--commit + REVACT_ALLOW_DESTRUCTIVE=1）+ 逐批批准</span></div>
      <details open><summary>探针注册表（${(probes.items || []).length} 个，按站点分组）与 live 标签</summary>
        ${siteBlocks}</details>
      <div class="panes"><div class="list" id="gp-list"></div><div class="detail" id="gp-detail"></div></div>`;
    $('#gp-mock').addEventListener('click', async () => {
      const rr = await API.runStage('probe', 'probe_mock', {});
      if (rr.ok && rr.job) { toast('mock 探针已启动', 'ok'); APP.showJob(rr.job.job_id); } else toast(rr.error, 'err');
    });
    $('#gp-live').addEventListener('click', async () => {
      const rr = await API.runStage('probe', 'probe_live', {});
      if (rr.ok && rr.job) { toast('live 探针已启动', 'ok'); APP.showJob(rr.job.job_id); } else toast(rr.error, 'err');
    });
    $('#gp-live-reddit').addEventListener('click', async () => {
      const rr = await API.runStage('probe', 'probe_live_reddit', {});
      if (rr.ok && rr.job) { toast('reddit live 探针已启动', 'ok'); APP.showJob(rr.job.job_id); } else toast(rr.error, 'err');
    });
    master($('#gp-list'), $('#gp-detail'), r.items.slice().reverse(),
      g => `${g.effective ? '●' : '○'} ${esc(g.action_type)} ${badge(g.label)}
        ${anns[g.probe_id] ? annBadge(anns[g.probe_id]) : ''}
        <span class="sub">${esc(g.timestamp || 'legacy row')}</span>`,
      g => {
        const a = anns[g.probe_id] || {};
        const undo = (g.undo_actions || []).map(x => `<li>${esc(x)}</li>`).join('');
        const shots = (g.shots || []).map(p => `<figure>
          <img src="/api/screenshot?path=${encodeURIComponent(p)}" alt="">
          <figcaption>${esc(p.split('/').pop())}</figcaption></figure>`).join('');
        return `<h3>${esc(g.action_type)} ${badge(g.label)}
            ${g.commit_mode ? badge('commit', 'IRREVERSIBLE') : badge('dry-run / non-destructive', 'plain')}
            ${g.effective ? badge('当前生效标签', 'EXECUTE') : ''}
            ${a.reversibility_override ? badge('人工覆核: ' + a.reversibility_override, 'needs-review') : ''}</h3>
          <dl class="kv"><dt>probe</dt><dd>${esc(g.probe_name || '(legacy)')}</dd>
            <dt>S→A→S′ 证据</dt><dd>baseline=${esc(JSON.stringify(g.evidence.baseline ?? '—'))}
              → after_action=${esc(JSON.stringify(g.evidence.after_action ?? g.evidence.after_add ?? '—'))}
              → after_undo=${esc(JSON.stringify(g.evidence.after_undo ?? '—'))}</dd>
            <dt>residual_diff (S″−S)</dt><dd>${esc(JSON.stringify(g.residual_diff ?? '—'))}</dd>
            <dt>undo_steps</dt><dd>${g.undo_steps ?? '—'}</dd>
            <dt>backend 信号</dt><dd>${esc(g.grounding)}</dd>
            <dt>probe_id / time</dt><dd>${esc(g.probe_id || '—')} · ${esc(g.timestamp || '—')}</dd></dl>
          ${kvTable(g.evidence)}
          ${undo ? `<h2>undo 轨迹（实测 ${g.undo_actions.length} 步，训练字段 undo_action 来源）</h2><ul class="undo">${undo}</ul>` : ''}
          ${shots ? `<h2>screenshots</h2><div class="shots">${shots}</div>` : ''}
          <h2>训练字段（该 run 派生）</h2>
          <pre>${esc(JSON.stringify({ undoable: g.label === 'REVERSIBLE', undo_action: (g.undo_actions || [])[0] || null, reversibility_label: g.label, grounding_evidence: g.grounding + ':' + JSON.stringify(g.evidence.baseline ?? null) }, null, 1))}</pre>
          ${reviewBox('grounded', g.probe_id || ('row-' + g.row), a, { extra: `
            <div class="fl"><label>标签覆核</label>
              <select class="ann-field" data-field="reversibility_override">${['', ...LABELS].map(l =>
                `<option${(a.reversibility_override || '') === l ? ' selected' : ''}>${l}</option>`).join('')}</select>
              <label>置信度</label><input type="number" class="ann-field" data-field="confidence" min="0" max="1" step="0.05" value="${a.confidence ?? ''}" style="width:80px"></div>
            <p class="warn-note">人工覆核不改写行为标签：与 pinned 标签冲突的样本在导出时被排除并记录审计（见导出说明）。</p>` })}`;
      });
  },
};

/* ------------------------------------------------------------- distill -- */
const DistillView = {
  async render(el) {
    const [tmplR, distR] = await Promise.all([
      API.get('/api/sft'), API.get('/api/sft?distilled=1')]);
    if (!tmplR.ok) { el.innerHTML = `<div class="empty">${esc(tmplR.error)}</div>`; return; }
    const anns = distR.annotations || {};
    const byId = {}; tmplR.items.forEach(s => { byId[s.sample_id] = s; });
    const items = distR.items || [];
    el.innerHTML = `
      <div class="fl">
        <label>limit</label><input type="number" id="ds-limit" value="10" min="1" style="width:90px">
        <button class="btn primary sm" id="ds-run">运行蒸馏（需 teacher key）</button>
        <span class="mini-note">已蒸馏 ${items.length} / ${tmplR.items.length}（teacher 在 pin 死结论下只写措辞；QC 拒绝矛盾输出）</span></div>
      <div class="panes"><div class="list" id="ds-list"></div><div class="detail" id="ds-detail"></div></div>`;
    $('#ds-run').addEventListener('click', async () => {
      const r = await API.runStage('distill', 'distill', { limit: $('#ds-limit').value });
      if (r.ok && r.job) { toast('蒸馏已启动', 'ok'); APP.showJob(r.job.job_id); } else toast(r.error, 'err');
    });
    master($('#ds-list'), $('#ds-detail'), items,
      d => `${esc(d.sample_id)} ${annBadge(anns[d.sample_id])}
        <span class="sub">${badge(d.decision)} ${badge(d.reversibility)}</span>`,
      d => {
        const t = byId[d.sample_id];
        const line = (tag, a, b) => `<tr><td>${tag}</td><td>${esc(a || '—')}</td><td>${esc(b || '—')}</td></tr>`;
        return `<h3>${esc(d.sample_id)} ${badge(d.decision)} ${badge(d.reversibility)} ${badge('prose: teacher', 'accepted')}</h3>
          <div class="goal">${esc(d.goal)}</div>
          <h2>条件蒸馏对比（结论 pin 死，只换措辞）</h2>
          <div class="tbl"><table><thead><tr><th>字段</th><th>模板 prose</th><th>teacher prose</th></tr></thead><tbody>
            ${line('observation', t && t.observation, d.observation)}
            ${line('reasoning', t && t.reasoning, d.reasoning)}
            ${line('prediction', t && t.prediction, d.prediction)}
            ${line('rev_check', t && t.rev_check, d.rev_check)}
          </tbody></table></div>
          <h2>pinned 结论（必须逐字一致）</h2>
          <dl class="kv"><dt>reversibility</dt><dd>${esc(d.reversibility)} ${t && t.reversibility === d.reversibility ? '✓' : '✗ 漂移!'}</dd>
            <dt>undo</dt><dd>${esc(d.undo)} ${t && t.undo === d.undo ? '✓' : '✗ 漂移!'}</dd>
            <dt>decision</dt><dd>${esc(d.decision)} ${t && t.decision === d.decision ? '✓' : '✗ 漂移!'}</dd>
            <dt>answer</dt><dd>${esc(d.answer)} ${t && t.answer === d.answer ? '✓' : '✗ 漂移!'}</dd></dl>
          <details><summary>完整 assistant 序列</summary>${seq(d.assistant)}</details>
          ${reviewBox('distill', d.sample_id, anns[d.sample_id])}`;
      });
    if (!items.length) {
      $('#ds-detail').innerHTML = `<div class="empty">尚无蒸馏产物。配置 teacher 模型 key 后点「运行蒸馏」，
        产物写入 train/sft/revact_sft_distilled.jsonl（不覆盖模板版）。</div>`;
    }
  },
};

/* ------------------------------------------------------------- quality -- */
const QualityView = {
  async render(el) {
    const r = await API.get('/api/quality');
    if (!r.ok) { el.innerHTML = `<div class="empty">${esc(r.error)}</div>`; return; }
    const q = r.quality, v = q.volumes, d = q.distributions;
    const lowRows = q.low_quality.map(x =>
      `<tr><td style="font-family:var(--mono)">${esc(x.sample_id)}</td><td>${esc(x.reason)}</td></tr>`).join('');
    el.innerHTML = `
      <div class="tiles">
        <div class="tile"><b>${v.sft_samples}</b><span>SFT 样本</span></div>
        <div class="tile"><b>${v.dpo_pairs}</b><span>DPO 对</span></div>
        <div class="tile"><b>${v.trajectories_success}/${v.trajectories}</b><span>成功轨迹（率 ${(q.rates.traj_success_rate * 100).toFixed(0)}%）</span></div>
        <div class="tile"><b>${v.key_states} / ${v.reached_states}</b><span>key / 风险状态</span></div>
        <div class="tile"><b>${v.grounded_action_classes}</b><span>grounded 动作类（${v.grounded_probe_runs} runs）</span></div>
        <div class="tile"><b>${(q.rates.distill_coverage * 100).toFixed(0)}%</b><span>蒸馏覆盖</span></div>
        <div class="tile"><b>${(q.teacher.pinned_label_agreement * 100).toFixed(0)}%</b><span>teacher-pinned 一致</span></div>
        <div class="tile"><b>${(q.counterfactual_coverage.coverage_rate * 100).toFixed(0)}%</b><span>反事实覆盖（均 ${q.counterfactual_coverage.avg_pairs_per_sample} 对/样本）</span></div>
        <div class="tile"><b>${q.n_low_quality}</b><span>低质量样本</span></div>
      </div>
      <div class="fl">
        <button class="btn sm" id="q-export">导出 JSONL/CSV（去数据集浏览器配置）</button>
        <button class="btn sm" id="q-viz">重建 HTML 报告（dataset_viz.html）</button>
        <button class="btn sm" id="q-refresh">重新计算</button></div>
      <h2>可逆 / 不可逆（生效标签）</h2>${bars(d.reversibility_effective)}
      <h2>oracle 决策分布</h2>${bars(d.decision)}
      <h2>约束风格分布</h2>${bars(d.constraint_style, () => 'plain')}
      <h2>DPO pair 类型</h2>${bars(d.pair_type, () => 'plain')}
      <h2>动作类分布</h2>${bars(d.action_type, () => 'plain')}
      <h2>split 分布</h2>${bars(d.split, k => k === 'test' ? 'test' : 'plain')}
      <h2>决策矩阵（action × variant → decision）</h2>
      <div class="tbl"><table><thead><tr><th>action</th><th>variant</th><th>decision</th><th>n</th></tr></thead>
        <tbody>${d.decision_matrix.map(m => `<tr><td>${esc(m.action_type)}</td><td>${esc(m.variant)}</td><td>${badge(m.decision)}</td><td>${m.n}</td></tr>`).join('')}</tbody></table></div>
      <h2>低质量样本（${q.n_low_quality}）</h2>
      <div class="tbl"><table><thead><tr><th>sample</th><th>原因</th></tr></thead>
        <tbody>${lowRows || '<tr><td colspan=2 class="empty">无</td></tr>'}</tbody></table></div>
      <h2>人工标注摘要</h2><pre>${esc(JSON.stringify(q.annotations, null, 1))}</pre>`;
    $('#q-refresh').addEventListener('click', async () => {
      await API.runStage('qc', 'compute', {}); this.render(el);
    });
    $('#q-export').addEventListener('click', () => APP.go('browser'));
    $('#q-viz').addEventListener('click', async () => {
      const rr = await API.runStage('export', 'viz', {});
      if (rr.ok && rr.job) { toast('viz 重建已启动', 'ok'); APP.showJob(rr.job.job_id); } else toast(rr.error, 'err');
    });
  },
};

/* -------------------------------------------------------------- browser -- */
const BrowserView = {
  f: { action_type: '', variant: '', style: '', decision: '', reversibility: '', split: '', status: '', q: '' },
  async render(el) {
    const [r, annR, exps] = await Promise.all([
      API.get('/api/sft'), API.get('/api/annotations/sample'), API.get('/api/exports')]);
    if (!r.ok) { el.innerHTML = `<div class="empty">${esc(r.error)}</div>`; return; }
    const anns = annR.ok ? annR.effective : {};
    const f = this.f;
    const items = r.items.filter(s => {
      const a = anns[s.sample_id] || {};
      return (!f.action_type || s.action_type === f.action_type)
        && (!f.variant || s.variant === f.variant)
        && (!f.style || s.constraint_style === f.style)
        && (!f.decision || s.decision === f.decision)
        && (!f.reversibility || s.reversibility === f.reversibility)
        && (!f.split || s.split === f.split)
        && (!f.status || (a.review_status || 'unreviewed') === f.status)
        && (!f.q || (s.sample_id + s.goal).toLowerCase().includes(f.q.toLowerCase()));
    });
    const sel = (k, vals) => `<select class="br-f" data-f="${k}">${selOpts(vals, f[k])}</select>`;
    el.innerHTML = `
      <div class="fl">
        <input type="text" class="br-f" data-f="q" placeholder="搜索 id / goal" value="${esc(f.q)}" style="min-width:180px">
        <label>action</label>${sel('action_type', r.items.map(s => s.action_type))}
        <label>variant</label>${sel('variant', r.items.map(s => s.variant))}
        <label>style</label>${sel('style', r.items.map(s => s.constraint_style))}
        <label>decision</label>${sel('decision', r.items.map(s => s.decision))}
        <label>rev</label>${sel('reversibility', r.items.map(s => s.reversibility))}
        <label>split</label>${sel('split', r.items.map(s => s.split))}
        <label>覆核</label>${sel('status', ['unreviewed', 'accepted', 'rejected', 'needs-review', 'confirmed'])}
        <span class="mini-note">${items.length} / ${r.items.length}</span></div>
      <details id="br-card"><summary>Dataset Card：样本完整形态与字段 schema（HF 风格）</summary>
        <div id="br-card-body" class="loading">加载中…</div></details>
      <details id="br-export"><summary>导出最终数据集（应用标注 overlay）</summary>
        <div class="fl" style="margin-top:8px">
          <label>名称</label><input type="text" id="ex-name" value="release" style="width:140px">
          <label>val 比例</label><input type="number" id="ex-val" value="0.15" step="0.05" min="0" max="0.5" style="width:80px">
          <label><input type="checkbox" id="ex-needs"> 含待复核</label>
          <label><input type="checkbox" id="ex-distill" checked> 优先蒸馏 prose</label>
          <button class="btn primary sm" id="ex-run">导出 train/val/test + dataset card</button></div>
        <div id="ex-result"></div>
        <h2>历史导出（${(exps.items || []).length}）</h2>
        <div class="tbl"><table><tbody>${(exps.items || []).map(x =>
          `<tr><td style="font-family:var(--mono)">${esc(x.name)}</td><td class="mini-note">${esc(x.files.join(', '))}</td></tr>`).join('') || '<tr><td class="empty">无</td></tr>'}</tbody></table></div>
      </details>
      <div class="panes"><div class="list" id="br-list"></div><div class="detail" id="br-detail"></div></div>`;
    $$('.br-f', el).forEach(x => x.addEventListener('change', e => {
      this.f[e.target.dataset.f] = e.target.value; this.render(el);
    }));
    $('#br-card').addEventListener('toggle', e => {
      if (e.target.open) this.renderCard();
    });
    $('#ex-run').addEventListener('click', async () => {
      const rr = await API.runStage('export', 'export', {
        name: $('#ex-name').value, val_frac: +$('#ex-val').value,
        include_needs_review: $('#ex-needs').checked,
        prefer_distilled: $('#ex-distill').checked,
      });
      if (rr.ok) {
        const res = rr.result;
        $('#ex-result').innerHTML = `<div class="goal">✓ ${esc(res.dir)}<br>
          train=${res.n_train} val=${res.n_val} test=${res.n_test} dpo=${res.n_dpo} 排除=${res.n_excluded}<br>
          文件：${esc(res.files.join(', '))}</div>`;
        toast('导出完成', 'ok');
      } else toast(rr.error, 'err');
    });
    master($('#br-list'), $('#br-detail'), items,
      s => `${esc(s.sample_id)} ${annBadge(anns[s.sample_id])}
        <span class="sub">${badge(s.decision)} ${badge(s.reversibility)} ${s.split === 'test' ? badge('test', 'test') : ''}</span>`,
      s => `<h3>${esc(s.sample_id)}</h3><div id="br-lineage" class="loading">加载 lineage…</div>`,
      s => this.loadLineage(s, anns));
  },

  async loadLineage(s, anns) {
    const [r, rawR] = await Promise.all([
      API.get('/api/lineage?sample=' + encodeURIComponent(s.sample_id)),
      API.get('/api/sample_raw?sample=' + encodeURIComponent(s.sample_id))]);
    const box = $('#br-lineage');
    if (!box) return;
    box.classList.remove('loading');
    if (!r.ok || !r.lineage) { box.innerHTML = `<div class="empty">${esc(r.error || '无 lineage')}</div>`; return; }
    const L = r.lineage, sm = L.sample;
    const chainStr = ['task/traj', 'key state', 'risk state', 'constraint', 'candidates',
      'undo label', 'teacher'].join(' → ');
    box.innerHTML = `
      <p class="mini-note">lineage: ${chainStr}</p>
      <dl class="kv">
        <dt>① 来源状态</dt><dd>${L.state ? esc(L.state.name) + ' (' + esc(L.state.source) + ', ' + esc(L.state.url) + ')' : '—'}</dd>
        <dt>② 相关 key states</dt><dd>${L.related_key_states.map(k => esc(k.state_id)).join(', ') || '—（reach 直达，非轨迹挖掘）'}</dd>
        <dt>③ 约束/目标</dt><dd>${esc(sm.goal)} ${badge(sm.constraint_style, 'plain')} <span class="mini-note">${esc(sm.goal_template)}</span></dd>
        <dt>④ grounded 标签</dt><dd>${badge(L.effective_label)}（${L.grounded_runs.length} 次探针 run）</dd>
        <dt>⑤ DPO 反事实</dt><dd>${L.dpo_pairs.map(p => badge(p.pair_type, 'plain')).join(' ') || '—'}</dd>
        <dt>⑥ teacher prose</dt><dd>${L.distilled ? badge('已蒸馏', 'accepted') : badge('模板', 'plain')}</dd></dl>
      <div class="goal">${esc(sm.goal)}</div>
      <h2>assistant 目标序列（${L.distilled ? '蒸馏版' : '模板版'}）</h2>
      ${seq((L.distilled || sm).assistant)}
      <details><summary>输入 observation（截断）</summary><pre>${esc(sm.obs)}</pre></details>
      ${L.dpo_pairs.length ? `<h2>DPO 对（chosen vs rejected）</h2>` + L.dpo_pairs.map(p => `
        <details><summary>${esc(p.pair_type)} · rejected 宣称 ${esc(p.rejected_reversibility)} / ${esc((p.rejected_decision || '').split(' ')[0])}</summary>
          <div class="pair"><div class="col ok"><h4>chosen</h4>${seq(p.chosen)}</div>
          <div class="col bad"><h4>rejected（${esc(p.pair_type)}）</h4>${seq(p.rejected)}</div></div></details>`).join('') : ''}
      ${this.rawBlock(rawR)}
      ${reviewBox('sample', s.sample_id, anns[s.sample_id])}`;
  },

  /* 完整样本形态：训练文件里的原始 JSONL 行，未截断（问「样本长什么样」看这里）。 */
  rawBlock(rawR) {
    if (!rawR.ok || !rawR.raw) return '';
    const raw = rawR.raw;
    const msg = m => `<div class="fcard">
      <h4>messages · ${badge(m.role, 'plain')} <span class="mini-note">${(m.content || '').length} chars</span></h4>
      <pre>${esc(m.content)}</pre></div>`;
    const dpoBlocks = (raw.dpo || []).map(p => `
      <details><summary>${esc(p.pair_id)}</summary><pre>${esc(JSON.stringify(p, null, 1))}</pre></details>`).join('');
    return `
      <h2>完整样本形态（原始 JSONL，未截断）</h2>
      <p class="mini-note">split=${esc(raw.split)} · 单步决策样本：输入 = system + user（goal + 当前
        observation，无历史轨迹），监督目标 = assistant 一条。字段 schema 见列表上方「Dataset Card」。</p>
      ${(raw.sft.messages || []).map(msg).join('')}
      <details><summary>meta（审计字段，训练不喂入）</summary>${kvTable(raw.sft.meta)}</details>
      <details><summary>原始 JSON（SFT 行，可直接对照 train/sft/revact_sft.jsonl）</summary>
        <pre>${esc(JSON.stringify(raw.sft, null, 1))}</pre></details>
      ${raw.distilled ? `<details><summary>蒸馏版原始 JSON（teacher prose，结论 pin 死）</summary>
        <pre>${esc(JSON.stringify(raw.distilled, null, 1))}</pre></details>` : ''}
      ${dpoBlocks ? `<h2>派生 DPO 对（原始 JSON，${raw.dpo.length} 条）</h2>${dpoBlocks}` : ''}`;
  },

  async renderCard() {
    const body = $('#br-card-body');
    if (!body) return;
    const r = await API.get('/api/dataset_card');
    body.classList.remove('loading');
    if (!r.ok) { body.innerHTML = `<div class="empty">${esc(r.error)}</div>`; return; }
    const c = r.card, sum = c.summary || {}, splits = sum.splits || {};
    const tbl = (head, rows) => `<div class="tbl"><table>
      <thead><tr>${head.map(h => `<th>${esc(h)}</th>`).join('')}</tr></thead>
      <tbody>${rows}</tbody></table></div>`;
    const mono = v => `<td style="font-family:var(--mono)">${esc(v)}</td>`;
    body.innerHTML = `
      <p class="note">${esc(c.granularity)}</p>
      <div class="tiles">
        <div class="tile"><b>${sum.n_sft ?? '—'}</b><span>SFT 样本（单步决策）</span></div>
        <div class="tile"><b>${sum.n_dpo ?? '—'}</b><span>DPO 偏好对</span></div>
        <div class="tile"><b>${Object.keys(sum.effective_labels || {}).length}</b><span>grounded 动作类</span></div>
        <div class="tile"><b>${splits.sft_train ?? '—'} / ${splits.sft_test ?? '—'}</b><span>split train / test</span></div>
        <div class="tile"><b>${sum.n_distilled ?? 0}</b><span>teacher 蒸馏样本</span></div>
      </div>
      <h2>一条 SFT 样本的三段 messages</h2>
      ${(c.message_flow || []).map(m => `<div class="fcard"><h4>${badge(m.role, 'plain')}</h4>
        <p class="note">${esc(m.desc)}</p></div>`).join('')}
      <h2>SFT 行字段 schema</h2>
      ${tbl(['字段', '类型', '说明'], c.sft_schema.map(f =>
        `<tr>${mono(f[0])}<td>${esc(f[1])}</td><td class="mini-note">${esc(f[2])}</td></tr>`).join(''))}
      <h2>meta 字段（审计 / 切分用，训练不喂入）</h2>
      ${tbl(['字段', '说明'], c.meta_schema.map(f =>
        `<tr>${mono(f[0])}<td class="mini-note">${esc(f[1])}</td></tr>`).join(''))}
      <h2>assistant 输出规格（&lt;think&gt; 七字段 + &lt;answer&gt;，格式 iris.v2）</h2>
      ${tbl(['字段', '内容'], c.assistant_format.map(f =>
        `<tr>${mono(f[0])}<td class="mini-note">${esc(f[1])}</td></tr>`).join(''))}
      <h2>DPO 行字段 schema</h2>
      ${tbl(['字段', '类型', '说明'], c.dpo_schema.map(f =>
        `<tr>${mono(f[0])}<td>${esc(f[1])}</td><td class="mini-note">${esc(f[2])}</td></tr>`).join(''))}
      <h2>消息长度统计（字符）</h2>
      ${tbl(['role', 'n', 'min', 'avg', 'max'], Object.entries(c.length_stats || {}).map(([k, v]) =>
        `<tr>${mono(k)}<td>${v.n ?? 0}</td><td>${v.min ?? '—'}</td><td>${v.avg ?? '—'}</td><td>${v.max ?? '—'}</td></tr>`).join(''))}
      <details><summary>system prompt（全数据集固定一条，prompt 指纹 ${esc(c.prompts_fingerprint || '—')}）</summary><pre>${esc(c.system_prompt)}</pre></details>
      <p class="mini-note">选中左侧任一样本，详情底部「完整样本形态」区块展示该样本未截断的原始 JSONL
        （messages 三段 + meta + 派生 DPO 对）。</p>`;
  },
};

/* ---------------------------------------------------------------- jobs -- */
const JobsView = {
  sel: null, timer: null,
  async render(el) {
    const r = await API.get('/api/jobs');
    if (!r.ok) { el.innerHTML = `<div class="empty">${esc(r.error)}</div>`; return; }
    el.innerHTML = `<div class="panes" style="grid-template-columns:360px 1fr">
      <div class="list" id="jb-list"></div><div class="detail" id="jb-detail"></div></div>`;
    master($('#jb-list'), $('#jb-detail'), r.jobs,
      j => `${stDot(j.status)}${esc(j.job_id)}
        <span class="sub">${esc(j.stage)}.${esc(j.action)} · ${fmtTime(j.started_at)}</span>`,
      j => `<h3>${stDot(j.status)}${esc(j.job_id)} <span class="badge b-plain">${esc(STATUS_ZH[j.status] || j.status)}</span></h3>
        <dl class="kv"><dt>命令</dt><dd>${esc((j.cmd || []).join(' ')) || '(in-process)'}</dd>
          <dt>env</dt><dd>${esc((j.env_names || []).join(', ')) || '—'}</dd>
          <dt>时间</dt><dd>${fmtTime(j.started_at)} → ${fmtTime(j.finished_at)}</dd>
          ${j.note ? `<dt>note</dt><dd>${esc(j.note)}</dd>` : ''}</dl>
        ${j.status === 'running' ? `<button class="btn danger sm" id="jb-stop">终止</button>` : ''}
        <h2>日志（key 已打码）</h2><div class="joblog" id="jb-log">加载中…</div>`,
      j => this.watch(j));
    if (this.sel) {
      const i = r.jobs.findIndex(j => j.job_id === this.sel);
      if (i >= 0) $$('#jb-list button')[i].click();
      this.sel = null;
    }
  },
  async watch(j) {
    clearInterval(this.timer);
    const load = async () => {
      const r = await API.get('/api/jobs/' + j.job_id);
      const logEl = $('#jb-log');
      if (!logEl) { clearInterval(this.timer); return; }
      if (r.ok) {
        logEl.textContent = r.log || '(空)';
        if (r.job.status !== 'running') clearInterval(this.timer);
      }
    };
    await load();
    this.timer = setInterval(load, 2000);
    const stop = $('#jb-stop');
    if (stop) stop.addEventListener('click', async () => {
      await API.post(`/api/jobs/${j.job_id}/stop`, {});
      toast('已发送终止'); APP.refreshCurrent();
    });
  },
};

const VIEWS = {
  pipeline: PipelineView, config: ConfigView, prompts: PromptsView, traj: TrajView,
  keystates: KeyStatesView, constraints: ConstraintsView,
  candidates: CandidatesView, grounded: GroundedView, distill: DistillView,
  quality: QualityView, browser: BrowserView, jobs: JobsView,
};
