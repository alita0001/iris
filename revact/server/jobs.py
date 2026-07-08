"""Subprocess job manager for pipeline stage runs.

Each run executes an existing CLI command (``python -m revact.cli ...``) in a
worker thread, streams merged stdout/stderr to
``outputs/workbench/jobs/<job_id>.log`` and records an index row in
``outputs/workbench/jobs.jsonl`` so history survives server restarts.

Secrets: values passed through ``env_extra`` (API keys) are exported to the
child environment only. They never appear in the command line or the index,
and log content is redacted against every secret value seen this session
before it is served.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .. import config

JOBS_DIR = config.OUTPUTS_DIR / "workbench" / "jobs"
INDEX_PATH = config.OUTPUTS_DIR / "workbench" / "jobs.jsonl"

_SECRET_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class JobManager:
    def __init__(self, jobs_dir: Path | None = None, index_path: Path | None = None):
        self.jobs_dir = jobs_dir or JOBS_DIR
        self.index_path = index_path or INDEX_PATH
        self._jobs: dict[str, dict] = {}
        self._procs: dict[str, subprocess.Popen] = {}
        self._secrets: set[str] = set()
        self._lock = threading.Lock()
        self._load_index()

    # ------------------------------------------------------------- index -- #
    def _load_index(self) -> None:
        if not self.index_path.exists():
            return
        for ln in self.index_path.open(encoding="utf-8"):
            ln = ln.strip()
            if not ln:
                continue
            try:
                row = json.loads(ln)
            except json.JSONDecodeError:
                continue
            row.setdefault("status", "unknown")
            if row["status"] == "running":       # stale from a dead server
                row["status"] = "interrupted"
            self._jobs[row["job_id"]] = row

    def _persist(self, job: dict) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with self.index_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(job, ensure_ascii=False) + "\n")

    # -------------------------------------------------------------- runs -- #
    def start(self, cmd: list[str], stage: str, action: str,
              env_extra: dict[str, str] | None = None,
              cwd: Path | None = None) -> dict:
        job_id = f"{stage}-{uuid.uuid4().hex[:8]}"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.jobs_dir / f"{job_id}.log"
        env = dict(os.environ)
        env_names = []
        for k, v in (env_extra or {}).items():
            if not v:
                continue
            env[k] = v
            env_names.append(k)
            if any(h in k.upper() for h in _SECRET_HINTS):
                self._secrets.add(v)
        job = {
            "job_id": job_id, "stage": stage, "action": action,
            "cmd": cmd, "env_names": env_names, "status": "running",
            "returncode": None, "started_at": _now(), "finished_at": None,
            "log_path": str(log_path),
        }
        with self._lock:
            self._jobs[job_id] = job
        self._persist(job)

        def worker():
            with log_path.open("w", encoding="utf-8") as logf:
                logf.write(f"$ {' '.join(cmd)}\n")
                if env_names:
                    logf.write(f"[env] {', '.join(env_names)} exported (values hidden)\n")
                logf.flush()
                try:
                    proc = subprocess.Popen(
                        cmd, cwd=str(cwd or config.PROJECT_ROOT), env=env,
                        stdout=logf, stderr=subprocess.STDOUT)
                    with self._lock:
                        self._procs[job_id] = proc
                    rc = proc.wait()
                except Exception as e:  # noqa: BLE001 - job must record any failure
                    logf.write(f"\n[workbench] spawn failed: {e}\n")
                    rc = -1
            with self._lock:
                job["status"] = "success" if rc == 0 else "failed"
                job["returncode"] = rc
                job["finished_at"] = _now()
                self._procs.pop(job_id, None)
            self._persist(job)

        threading.Thread(target=worker, daemon=True).start()
        return job

    def record_instant(self, stage: str, action: str, ok: bool, note: str) -> dict:
        """Index an in-process (non-subprocess) action so it shows in history."""
        job = {
            "job_id": f"{stage}-{uuid.uuid4().hex[:8]}", "stage": stage,
            "action": action, "cmd": [], "env_names": [],
            "status": "success" if ok else "failed", "returncode": 0 if ok else 1,
            "started_at": _now(), "finished_at": _now(),
            "log_path": "", "note": note,
        }
        with self._lock:
            self._jobs[job["job_id"]] = job
        self._persist(job)
        return job

    def stop(self, job_id: str) -> bool:
        with self._lock:
            proc = self._procs.get(job_id)
        if proc is None:
            return False
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        return True

    # ------------------------------------------------------------- reads -- #
    def list(self) -> list[dict]:
        with self._lock:
            return sorted(self._jobs.values(),
                          key=lambda j: j["started_at"], reverse=True)

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            return self._jobs.get(job_id)

    def log_tail(self, job_id: str, max_chars: int = 20000) -> str:
        job = self.get(job_id)
        if not job or not job.get("log_path"):
            return ""
        p = Path(job["log_path"])
        if not p.exists():
            return ""
        text = p.read_text(encoding="utf-8", errors="replace")
        if len(text) > max_chars:
            text = f"… (+{len(text) - max_chars} chars)\n" + text[-max_chars:]
        for secret in self._secrets:
            if secret:
                text = text.replace(secret, "***")
        return text

    def last_for_stage(self, stage: str) -> dict | None:
        for job in self.list():
            if job["stage"] == stage:
                return job
        return None


MANAGER = JobManager()


def python_cli(*args: str) -> list[str]:
    """Command prefix reusing the current interpreter + existing CLI."""
    return [sys.executable, "-m", "revact.cli", *args]


def wait_all(timeout: float = 60.0) -> None:
    """Test helper: block until no job is running."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if all(j["status"] != "running" for j in MANAGER.list()):
            return
        time.sleep(0.1)
