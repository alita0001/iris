"""Workbench HTTP server (stdlib http.server — the package stays dependency-free).

Routes (JSON unless noted):
  GET  /                                  ui/index.html (static frontend)
  GET  /ui/<file>                         static assets
  GET  /api/health                        summary + pipeline + env readiness
  GET  /api/pipeline                      stage cards with status/artifacts
  POST /api/pipeline/run                  {stage, action, params} -> job/result
  GET  /api/jobs                          job history
  GET  /api/jobs/<id>                     job + redacted log tail
  POST /api/jobs/<id>/stop
  GET  /api/config                        settings (secrets masked)
  POST /api/config                        update settings; secrets -> memory only
  POST /api/config/save                   persist settings (secrets stripped)
  GET  /api/prompts                       prompt registry (defaults + overrides)
  POST /api/prompts                       {id, value} save prompt override
  POST /api/prompts/reset                 {id} drop override, back to default
  GET  /api/trajectories                  index; /api/trajectories/<id> detail
  GET  /api/keystates | /api/states | /api/grounded | /api/probes
  GET/POST /api/probe-specs               label-free declarative probe specs
  GET  /api/sft[?distilled=1] | /api/dpo | /api/templates | /api/quality
  GET  /api/dataset_card                  sample anatomy + schema + counts
  GET  /api/sample_raw?sample=<sample_id> full unclipped SFT/distilled/DPO rows
  GET  /api/constraints/preview[?state=]  real build_goal output
  GET  /api/candidates?state=<name>       expert/safe/DPO counterfactuals
  GET  /api/lineage?sample=<sample_id>
  GET  /api/annotations/<kind>            effective + history
  POST /api/annotations                   {kind, target_id, payload}
  GET  /api/exports                       past exports
  GET  /api/screenshot?path=<rel>         PNG under data/ (path-checked)

Security: binds 127.0.0.1 by default; API-key values are accepted via POST
/api/config, kept in RUNTIME.secrets (memory), masked in every GET, stripped
from the persisted file, and exported only into job child environments.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .. import config, prompts
from .. import prompt_store
from ..grounding.authoring import (load_authored_specs, save_authored_spec,
                                   spec_from_workbench)
from ..grounding.schema import (EFFECT_STATUSES, GROUNDING_SCHEMA_VERSION,
                                RECOVERY_STATUSES)
from . import annotations
from .adapters import RUNTIME, live_ready, pipeline_overview, run_action
from .datasets import DataStore
from .export import EXPORTS_DIR
from .jobs import MANAGER

UI_DIR = Path(__file__).resolve().parent.parent / "ui"
LOCAL_CONFIG_PATH = config.CONFIG_DIR / "workbench.local.json"

_MIME = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
         ".js": "text/javascript; charset=utf-8", ".png": "image/png",
         ".svg": "image/svg+xml", ".json": "application/json"}

_lock = threading.Lock()


def load_local_config() -> None:
    """Merge persisted (secret-free) settings into RUNTIME at startup."""
    if not LOCAL_CONFIG_PATH.exists():
        return
    try:
        saved = json.loads(LOCAL_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    _deep_update(RUNTIME.settings, saved if isinstance(saved, dict) else {})


def _deep_update(dst: dict, src: dict) -> None:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        elif k not in ("api_key",):          # never accept persisted key values
            dst[k] = v


def _masked_settings() -> dict:
    s = json.loads(json.dumps(RUNTIME.settings))   # deep copy
    for role, m in (s.get("models") or {}).items():
        key_env = m.get("api_key_env", "")
        m["api_key_set"] = bool(RUNTIME.secrets.get(key_env))
        m.pop("api_key", None)
    return s


def _apply_config(body: dict) -> None:
    models = body.get("models") or {}
    for role, m in models.items():
        if not isinstance(m, dict):
            continue
        key_val = m.pop("api_key", None)
        key_env = m.get("api_key_env") or \
            ((RUNTIME.settings.get("models") or {}).get(role) or {}) \
            .get("api_key_env", "")
        if key_val and key_env:
            RUNTIME.secrets[key_env] = str(key_val)
        if key_val == "":                     # explicit clear
            RUNTIME.secrets.pop(key_env, None)
    _deep_update(RUNTIME.settings, {k: v for k, v in body.items()})


def save_config() -> dict:
    """Persist settings WITHOUT secrets to configs/workbench.local.json."""
    data = _masked_settings()
    for m in (data.get("models") or {}).values():
        m.pop("api_key_set", None)
    LOCAL_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_CONFIG_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"ok": True, "path": str(LOCAL_CONFIG_PATH),
            "note": "已保存（不含任何 key 值；key 只存在服务进程内存）"}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "iris-workbench/0.1"

    # --------------------------------------------------------------- util -- #
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _err(self, code: int, msg: str) -> None:
        self._json({"ok": False, "error": msg}, code)

    def _body(self) -> dict:
        try:
            n = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(n).decode("utf-8")) if n else {}
        except (ValueError, json.JSONDecodeError):
            return {}

    def log_message(self, fmt, *args):   # quiet default logging
        pass

    # ---------------------------------------------------------------- GET -- #
    def do_GET(self):  # noqa: N802 (http.server API)
        u = urlparse(self.path)
        path, q = u.path.rstrip("/") or "/", parse_qs(u.query)
        store = DataStore()
        try:
            if path == "/" or path == "/index.html":
                return self._static(UI_DIR / "index.html")
            if path.startswith("/ui/"):
                return self._static(UI_DIR / path[len("/ui/"):])
            if path == "/api/health":
                return self._json({
                    "ok": True, "summary": store.summary(),
                    "live_ready": live_ready(),
                    "n_running": sum(1 for j in MANAGER.list()
                                     if j["status"] == "running"),
                    "last_job": (MANAGER.list() or [None])[0],
                    "outputs_dir": str(config.OUTPUTS_DIR),
                    "viz_report": str(config.OUTPUTS_DIR / "dataset_viz.html"),
                })
            if path == "/api/pipeline":
                return self._json({"ok": True, "stages": pipeline_overview(store)})
            if path == "/api/jobs":
                return self._json({"ok": True, "jobs": MANAGER.list()[:100]})
            if path.startswith("/api/jobs/"):
                jid = path.split("/")[3]
                job = MANAGER.get(jid)
                if not job:
                    return self._err(404, f"no job {jid}")
                return self._json({"ok": True, "job": job,
                                   "log": MANAGER.log_tail(jid)})
            if path == "/api/config":
                return self._json({"ok": True, "settings": _masked_settings(),
                                   "local_config": str(LOCAL_CONFIG_PATH),
                                   "local_config_exists": LOCAL_CONFIG_PATH.exists()})
            if path == "/api/prompts":
                return self._json({"ok": True, "items": prompts.registry_view(),
                                   "fingerprint": prompts.fingerprint(),
                                   "overrides_file": str(prompts.overrides_path()),
                                   "bundle_dir": str(prompt_store.bundle_dir())})
            if path == "/api/prompts/bundle":
                fp = q.get("fp", [""])[0]
                try:
                    return self._json({"ok": True,
                                       "bundle": prompt_store.load_bundle(fp)})
                except (FileNotFoundError, ValueError):
                    return self._err(404, "prompt bundle missing or invalid")
            if path == "/api/prompts/diff":
                left, right = q.get("left", [""])[0], q.get("right", [""])[0]
                try:
                    return self._json({"ok": True, **prompt_store.diff_bundles(left, right)})
                except (FileNotFoundError, ValueError):
                    return self._err(404, "prompt bundle missing or invalid")
            if path == "/api/trajectories":
                return self._json({"ok": True, "items": store.trajectory_index()})
            if path.startswith("/api/trajectories/"):
                t = store.trajectory(path.split("/", 3)[3])
                return self._json({"ok": True, "trajectory": t}) if t \
                    else self._err(404, "no such trajectory")
            if path == "/api/keystates":
                return self._json({"ok": True, "items": store.key_states(),
                                   "annotations": annotations.effective("key_state")})
            if path == "/api/states":
                return self._json({"ok": True, "items": store.reached_states(),
                                   "annotations": annotations.effective("state")})
            if path == "/api/grounded":
                formal = store.formal_grounding()
                return self._json({"ok": True,
                                   # Compatibility aliases; explicitly legacy.
                                   "items": store.grounded_runs(),
                                   "effective_labels": store.effective_labels(),
                                   "manifest": store.manifest(),
                                   "legacy_class_smoke": {
                                       "items": store.grounded_runs(),
                                       "effective_labels": store.effective_labels(),
                                       "manifest": store.manifest(),
                                       "formal_supervision": False,
                                   },
                                   "formal_point": formal,
                                   "canonical_schema": {
                                       "schema_version": GROUNDING_SCHEMA_VERSION,
                                       "effect_status": list(EFFECT_STATUSES),
                                       "recovery_status": list(RECOVERY_STATUSES),
                                       "legacy_irreversible_is_display_only": True,
                                   },
                                   "annotations": annotations.effective("grounded")})
            if path == "/api/probes":
                return self._json({"ok": True, "items": store.probe_specs()})
            if path == "/api/probe-specs":
                spec_path = (store.root / "grounded" / "probe_specs" /
                             "authored_specs.jsonl")
                return self._json({
                    "ok": True,
                    "items": [spec.to_dict() for spec in
                              load_authored_specs(spec_path)],
                    "execution_enabled": False,
                    "label_entry_supported": False,
                    "note": "Specs define action/signal/undo/solver/budget only; labels are execution outputs.",
                })
            if path == "/api/sft":
                distilled = q.get("distilled", ["0"])[0] == "1"
                family = q.get("family", ["single"])[0]
                tier = q.get("tier", ["legacy"])[0]
                return self._json({"ok": True,
                                   "items": store.sft(distilled=distilled,
                                                      family=family,
                                                      tier=tier),
                                   "asset_tier": tier,
                                   "annotations": annotations.effective(
                                       "distill" if distilled else "sample")})
            if path == "/api/dpo":
                return self._json({"ok": True, "items": store.dpo(
                    family=q.get("family", ["single"])[0],
                    tier=q.get("tier", ["legacy"])[0]),
                    "asset_tier": q.get("tier", ["legacy"])[0]})
            if path == "/api/dataset_card":
                return self._json({"ok": True, "card": store.dataset_card()})
            if path == "/api/sample_raw":
                sid = q.get("sample", [""])[0]
                raw = store.sample_raw(
                    sid, tier=q.get("tier", ["auto"])[0]) if sid else None
                return self._json({"ok": bool(raw), "raw": raw})
            if path == "/api/templates":
                return self._json({"ok": True, **store.constraint_templates(),
                                   "annotations": annotations.effective("constraint")})
            if path == "/api/constraints/preview":
                name = q.get("state", [""])[0]
                if name:
                    p = store.constraint_preview(name)
                    return self._json({"ok": bool(p), "preview": p})
                previews = [p for s in store.reached_states()
                            if (p := store.constraint_preview(s["name"]))]
                return self._json({"ok": True, "previews": previews})
            if path == "/api/candidates":
                name = q.get("state", [""])[0]
                res = store.candidates_for(name) if name else None
                anns = annotations.effective("candidate")
                mine = {k: v for k, v in anns.items()
                        if k == name or k.startswith(name + "__")}
                return self._json({"ok": bool(res), "candidates": res,
                                   "annotations": mine})
            if path == "/api/lineage":
                sid = q.get("sample", [""])[0]
                lin = store.lineage(sid) if sid else None
                if lin:
                    lin["annotations"] = {
                        "sample": annotations.effective("sample").get(sid),
                        "distill": annotations.effective("distill").get(sid),
                    }
                return self._json({"ok": bool(lin), "lineage": lin})
            if path == "/api/quality":
                from .quality import compute_quality
                return self._json({"ok": True, "quality": compute_quality(store)})
            if path.startswith("/api/annotations/"):
                kind = path.split("/")[3]
                if kind not in annotations.KINDS:
                    return self._err(404, f"unknown kind {kind}")
                return self._json({"ok": True,
                                   "effective": annotations.effective(kind),
                                   "history": annotations.history(kind)[-500:]})
            if path == "/api/exports":
                items = []
                if EXPORTS_DIR.exists():
                    for d in sorted(EXPORTS_DIR.iterdir(), reverse=True):
                        if d.is_dir():
                            items.append({"name": d.name,
                                          "files": sorted(p.name for p in d.iterdir())})
                return self._json({"ok": True, "dir": str(EXPORTS_DIR),
                                   "items": items[:20]})
            if path == "/api/screenshot":
                return self._screenshot(q.get("path", [""])[0], store)
            return self._err(404, f"no route {path}")
        except BrokenPipeError:
            pass
        except Exception as e:  # noqa: BLE001 - convert to JSON error, keep serving
            self._err(500, f"{type(e).__name__}: {e}")

    def _static(self, p: Path) -> None:
        p = p.resolve()
        if UI_DIR not in p.parents and p != UI_DIR / "index.html":
            return self._err(403, "forbidden")
        if not p.is_file():
            return self._err(404, "not found")
        self._send(200, p.read_bytes(),
                   _MIME.get(p.suffix, "application/octet-stream"))

    def _screenshot(self, rel: str, store: DataStore) -> None:
        if not rel:
            return self._err(400, "path required")
        p = (store.root / rel).resolve()
        if store.root.resolve() not in p.parents:
            return self._err(403, "path outside data root")
        if not p.is_file() or p.suffix.lower() != ".png":
            return self._err(404, "no such screenshot")
        self._send(200, p.read_bytes(), "image/png")

    # --------------------------------------------------------------- POST -- #
    def do_POST(self):  # noqa: N802 (http.server API)
        path = urlparse(self.path).path.rstrip("/")
        body = self._body()
        store = DataStore()
        try:
            if path == "/api/pipeline/run":
                with _lock:
                    res = run_action(str(body.get("stage", "")),
                                     str(body.get("action", "")),
                                     body.get("params") or {})
                return self._json(res, 200 if res.get("ok") else 400)
            if path == "/api/config":
                with _lock:
                    _apply_config(body)
                return self._json({"ok": True, "settings": _masked_settings()})
            if path == "/api/config/save":
                with _lock:
                    _apply_config(body) if body else None
                    return self._json(save_config())
            if path == "/api/prompts":
                with _lock:
                    try:
                        before = prompts.effective()
                        before_fp = prompts.fingerprint()
                        prompt_store.store_bundle(before, author="workbench-before-edit")
                        prompts.set_override(str(body.get("id", "")),
                                             body.get("value"))
                        prompt_store.store_bundle(
                            prompts.effective(), parent_fp=before_fp,
                            author=str(body.get("author", "workbench")))
                    except ValueError as e:
                        return self._err(400, str(e))
                return self._json({"ok": True,
                                   "fingerprint": prompts.fingerprint(),
                                   "note": "已保存覆盖；改动 agent_system/模板池后"
                                           "需重跑 assemble（+multiturn+split）重物化样本"})
            if path == "/api/prompts/reset":
                with _lock:
                    before_fp = prompts.fingerprint()
                    prompt_store.store_bundle(prompts.effective(),
                                               author="workbench-before-reset")
                    prompts.clear_override(str(body.get("id", "")))
                    prompt_store.store_bundle(
                        prompts.effective(), parent_fp=before_fp,
                        author=str(body.get("author", "workbench")))
                return self._json({"ok": True,
                                   "fingerprint": prompts.fingerprint()})
            if path == "/api/probe-specs":
                with _lock:
                    proposal = dict(body.get("proposal") or {})
                    proposal.setdefault("author", str(body.get("author") or
                                                       "workbench"))
                    spec = spec_from_workbench(
                        proposal,
                        timestamp=datetime.now(timezone.utc).isoformat(
                            timespec="seconds"),
                        controller_version=config.CONTROLLER_VERSION,
                    )
                    artifact = (store.root / "grounded" / "probe_specs" /
                                "authored_specs.jsonl")
                    save_authored_spec(spec, artifact)
                return self._json({"ok": True, "spec": spec.to_dict(),
                                   "artifact": str(artifact),
                                   "note": "spec pending fixture + code review; no label created"})
            if path == "/api/annotations":
                kind = str(body.get("kind", ""))
                row = annotations.add(kind, str(body.get("target_id", "")),
                                      body.get("payload") or {},
                                      author=str(body.get("author", "workbench")))
                return self._json({"ok": True, "annotation": row})
            if path.startswith("/api/jobs/") and path.endswith("/stop"):
                jid = path.split("/")[3]
                return self._json({"ok": MANAGER.stop(jid)})
            return self._err(404, f"no route {path}")
        except (ValueError, KeyError) as e:
            self._err(400, str(e))
        except Exception as e:  # noqa: BLE001
            self._err(500, f"{type(e).__name__}: {e}")


def serve(host: str = "127.0.0.1", port: int = 7788) -> int:
    load_local_config()
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"[workbench] IRIS dataset workbench -> http://{host}:{port}")
    print(f"[workbench] data root: {config.DATA_ROOT}")
    print("[workbench] API keys: memory-only; saved config strips secrets.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[workbench] bye")
    return 0
