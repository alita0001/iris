/* Thin fetch wrappers for the workbench API. */
const API = {
  async get(path) {
    const r = await fetch(path);
    const j = await r.json().catch(() => ({ ok: false, error: `bad json (${r.status})` }));
    if (!r.ok && j.error === undefined) j.error = `HTTP ${r.status}`;
    return j;
  },
  async post(path, body) {
    const r = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body || {}),
    });
    const j = await r.json().catch(() => ({ ok: false, error: `bad json (${r.status})` }));
    if (!r.ok && j.error === undefined) j.error = `HTTP ${r.status}`;
    return j;
  },
  annotate(kind, targetId, payload) {
    return this.post('/api/annotations', { kind, target_id: targetId, payload });
  },
  runStage(stage, action, params) {
    return this.post('/api/pipeline/run', { stage, action, params: params || {} });
  },
};
