"""Local workbench server: read-only data APIs + pipeline adapters + overlays.

Design contract (keeps the existing pipeline untouched):
  * pipeline artifacts under ``data/`` are READ-ONLY for this package;
  * every human edit is an overlay row in ``data/annotations/*.jsonl``
    (see ``annotations.py``) joined by natural keys, applied only at export;
  * pipeline stages run through the existing CLI as subprocesses
    (``adapters.py`` + ``jobs.py``) — no business logic is duplicated;
  * API keys live in process memory only; persisted config keeps just the
    env-var NAME (see ``app.py``).
"""
