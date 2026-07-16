/* App shell: tab routing, health polling, job indicator. */
const APP = {
  current: 'pipeline',

  async go(tab) {
    this.current = tab;
    $$('#nav button').forEach(b => b.classList.toggle('on', b.dataset.t === tab));
    $$('section').forEach(s => s.classList.toggle('on', s.id === 'tab-' + tab));
    await this.refreshCurrent();
  },

  async refreshCurrent() {
    const el = $('#v-' + this.current);
    const view = VIEWS[this.current];
    if (!el || !view) return;
    el.classList.remove('loading');
    try {
      await view.render(el);
    } catch (e) {
      el.innerHTML = `<div class="empty">渲染失败：${esc(e.message)}</div>`;
      console.error(e);
    }
  },

  showJob(jobId) {
    JobsView.sel = jobId;
    this.go('jobs');
  },

  async pollHealth() {
    try {
      const h = await API.get('/api/health');
      if (h.ok) {
        const s = h.summary;
        $('#health-meta').textContent =
          `${s.site} · ${s.n_traj} 轨迹 / formal ${s.n_grounded_points} points / ` +
          `${s.n_sft} SFT / ${s.n_dpo} DPO · ` +
          `${h.live_ready ? 'live 就绪' : 'offline'} · ${h.n_running} 任务运行中`;
        $('#jobs-dot').hidden = !h.n_running;
      } else {
        $('#health-meta').textContent = '服务异常: ' + h.error;
      }
    } catch (e) {
      $('#health-meta').textContent = '服务连接失败';
    }
  },
};

$$('#nav button').forEach(b => b.addEventListener('click', () => APP.go(b.dataset.t)));
APP.pollHealth();
setInterval(() => APP.pollHealth(), 5000);
APP.refreshCurrent();
