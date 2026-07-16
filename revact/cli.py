"""Unified CLI:  python -m revact.cli <command>   (or `revact <command>`).

Commands (offline ones need no env/key; live ones need WA_* + conda agentlab):

  collect    roll out a policy, log trajectories + key states     [live|--mock]
  reach      deterministically reach pilot risk states            [live]
  crawl      harvest product URLs from catalog search             [live]
  scale      build many risk-affording states across products     [live]
  probe      run grounding probes (list with --list)              [live|--mock]
  assemble   S6-S8: grounded states + labels -> SFT/DPO           [offline]
  split      product-level train/test split                       [offline]
  train      LoRA SFT (--dry-run validates data/config only)      [gpu|--dry-run]
  eval       decision accuracy on held-out split                  [gpu|--dry-run]
  distill    S7 teacher prose distillation (pinned conclusions)   [needs key]
  collect-opinions  label-blind pre-action opinion baseline       [needs key]
  preflight-episode-source  verify raw calls into a review sheet  [offline]
  inspect    dataset quality stats (trajectories, key states)     [offline]
  viz        build the self-contained dataset explorer HTML       [offline]
  serve      dataset construction workbench (local web UI + API)  [offline]

Destructive probes additionally require BOTH --commit and env
REVACT_ALLOW_DESTRUCTIVE=1; without them they run dry (never click the
mutating control). API keys are only ever read from environment variables.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import sys
from pathlib import Path

from . import config


# --------------------------------------------------------------------------- #
# Live-env helpers (lazy imports keep offline commands dependency-free)
# --------------------------------------------------------------------------- #
def _make_live_renv(task_id: str | None = None, headless: bool = True,
                    save_screenshots: bool = False, site: str = "shopping"):
    from .envs.harness import RevActEnv, make_env
    if task_id is None:
        spec = config.SITES.get(site)
        task_id = spec.session_task if spec else config.SESSION_TASK_ID
    env = make_env(task_id, headless=headless)
    return RevActEnv(env, task_id="cli", site=site,
                     save_screenshots=save_screenshots)


def _make_mock_renv(goal: str = "mock session", site: str = "shopping"):
    from .envs.harness import RevActEnv
    from .envs.mock_env import MockRedditEnv, MockShoppingEnv
    env = MockRedditEnv(goal=goal) if site == "reddit" else MockShoppingEnv(goal=goal)
    return RevActEnv(env, task_id=f"mock-{site}", site=site)


def _probe_ctx(renv, site: str, base: str, product: str, commit: bool,
               budget: int, submission_url: str = "", forum_url: str = ""):
    """Build a site-appropriate ProbeContext (reddit needs submission/forum urls)."""
    from .grounding import ProbeContext
    kwargs = dict(renv=renv, base=base, commit=commit, budget=budget,
                  admin_base=config.WA_SHOPPING_ADMIN)
    if site == "reddit":
        kwargs["forum_url"] = forum_url or (base.rstrip("/") + "/f/books")
        # a stable books submission in the WebArena mirror; override per-mirror
        kwargs["submission_url"] = submission_url or (base.rstrip("/") + "/f/books/59421")
    else:
        kwargs["product_url"] = product
    return ProbeContext(**kwargs)


_FORMAL_PROBE_SPEC_FIELDS = {
    "probe_point_id", "probe_run_id", "state_id", "candidate_id",
    "candidate_snapshot_hash",
    "action_instance_id", "raw_action", "canonical_action",
    "environment_family", "environment_instance", "environment_origin",
    "is_mock", "task_id", "trajectory_id", "run_id", "seed", "url",
    "account", "privilege", "solver_set", "code_version", "product_url",
    "submission_url", "forum_url", "budget",
}
def _load_formal_probe_specs(path: Path) -> dict[str, dict]:
    """Load only reviewed ``iris.probe_execution_spec.v1`` contracts."""
    from .grounding.authoring import load_probe_execution_specs
    return {spec.probe_name: spec.to_probe_row()
            for spec in load_probe_execution_specs(path)}


def _default_product_url() -> str | None:
    import glob
    for f in sorted(glob.glob(str(config.RAW_TRAJ_DIR / "*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            try:
                url = json.loads(line).get("url_after", "")
            except json.JSONDecodeError:
                continue
            if url.endswith(".html") and config.WA_SHOPPING \
                    and url.startswith(config.WA_SHOPPING):
                return url
    if config.PRODUCT_URLS_PATH.exists():
        urls = json.loads(config.PRODUCT_URLS_PATH.read_text())
        if urls:
            return urls[0]
    return None


# --------------------------------------------------------------------------- #
# probe
# --------------------------------------------------------------------------- #
def cmd_probe(args) -> int:
    from .grounding import (list_probes, run_probe, save_formal_probe_results,
                            save_results)
    from .grounding import probes  # noqa: F401  (registers all probes)

    try:
        formal_specs = (_load_formal_probe_specs(Path(args.formal_spec))
                        if args.formal_spec else {})
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"ERROR: invalid --formal-spec: {exc}")
        return 1

    if args.list:
        print(f"{'name':34s} {'site':16s} {'destructive':16s} "
              f"{'grounding':22s} expected")
        for s in list_probes(site=args.site or None):
            print(f"{s.name:34s} {s.site:16s} {s.destructive:16s} "
                  f"{s.grounding:22s} {s.expected_spectrum}")
        return 0

    names = args.names
    if args.all_nondestructive:
        names = [s.name for s in list_probes(site=args.site or None)
                 if s.destructive in ("non_destructive", "self_recovering")]
    elif args.site and not names:
        names = [s.name for s in list_probes(site=args.site)]
    if not names:
        print("ERROR: give probe names, --all-nondestructive, --site, or --list")
        return 1
    if formal_specs and set(names) != set(formal_specs):
        print("ERROR: requested probe names must exactly match --formal-spec "
              f"rows; requested={sorted(names)} specs={sorted(formal_specs)}")
        return 1

    # group requested probes by their site so each site gets the right env
    by_site: dict[str, list[str]] = {}
    for name in names:
        try:
            site = get_probe_site(name)
        except KeyError as e:
            print(f"ERROR: {e}")
            return 1
        by_site.setdefault(site, []).append(name)

    results = []
    formal_runs = []
    for site, site_names in by_site.items():
        if args.mock:
            renv = _make_mock_renv(site=site)
            base = f"http://mock-{site}"
            product = "http://mock-shopping/product/20"
        else:
            base = config.site_base(site)
            if not base:
                spec = config.SITES.get(site)
                env_name = spec.base_env if spec else "WA_*"
                print(f"ERROR: {env_name} not set for site {site!r} "
                      f"(source scripts/export_webarena_env.sh)")
                return 1
            product = ""
            if site == "shopping":
                product = args.product_url or _default_product_url()
                if not product:
                    print("ERROR: no product url (pass --product-url or run `revact crawl`)")
                    return 1
            renv = _make_live_renv(headless=not args.no_headless,
                                   save_screenshots=args.screenshots, site=site)
        try:
            renv.reset(seed=0)
            for name in site_names:
                if renv.save_screenshots:
                    stamp = __import__("time").strftime("%Y%m%d-%H%M%S")
                    renv.trajectory_id = f"probe_{name.replace('.', '_')}_{stamp}"
                ctx = _probe_ctx(renv, site, base, product, args.commit, args.budget,
                                 submission_url=args.submission_url,
                                 forum_url=args.forum_url)
                if formal_specs:
                    spec = formal_specs[name]
                    registered = next(s for s in list_probes() if s.name == name)
                    if spec.get("site") and spec["site"] != registered.site:
                        print(f"ERROR: {name}: spec site != registered site")
                        return 1
                    if (spec.get("action_type") and
                            spec["action_type"] != registered.action_type):
                        print(f"ERROR: {name}: spec action_type != registered action_type")
                        return 1
                    if spec.get("requires_commit_flag") and not args.commit:
                        print(f"ERROR: {name}: reviewed destructive spec requires --commit")
                        return 1
                    overrides = {key: value for key, value in spec.items()
                                 if key in _FORMAL_PROBE_SPEC_FIELDS}
                    ctx = replace(ctx, **overrides)
                r = run_probe(name, ctx)
                if formal_specs:
                    spec = formal_specs[name]
                    observed_channels = set(
                        (r.evidence or {}).get("signal_channels") or [])
                    required_channels = set(spec.get("signal_channels") or [])
                    if not required_channels <= observed_channels:
                        print(f"ERROR: {name}: executed probe evidence is missing "
                              f"reviewed signal channels {sorted(required_channels - observed_channels)}")
                        return 1
                    observed_undo = tuple((r.evidence or {}).get("undo_actions") or [])
                    authored_undo = {tuple(seq) for seq in spec.get("undo_sequences") or []}
                    if r.recovery_status in {"RECOVERED", "PARTIALLY_RECOVERED"} \
                            and observed_undo not in authored_undo:
                        print(f"ERROR: {name}: executed undo trace does not match any "
                              "reviewed undo sequence")
                        return 1
                    r.evidence["authored_spec_id"] = spec.get("authored_spec_id")
                if renv.save_screenshots:
                    r.evidence["screenshots_dir"] = str(
                        Path("raw") / "screenshots" / renv.trajectory_id)
                results.append(r)
                if formal_specs:
                    formal_runs.append((r, ctx))
                ev = {k: v for k, v in r.evidence.items() if k != "undo_actions"}
                print(f"  [{site}] {name:30s} -> {r.label:22s} {ev}")
        finally:
            renv.close()

    if formal_specs:
        try:
            body, manifest = save_formal_probe_results(formal_runs)
        except Exception as exc:  # schema gate is intentionally fail closed
            print(f"ERROR: formal point persistence rejected: {exc}")
            return 1
        print(f"formal points -> {body}\nformal manifest -> {manifest}")
    elif not args.mock:
        path = save_results(results)
        print(f"saved -> {path}")
    return 0


def get_probe_site(name: str) -> str:
    from .grounding import get_probe
    return get_probe(name).site


def cmd_collect_points(args) -> int:
    """Deterministically reach reproducible decision states (full lineage)."""
    from .data.point_collect import (collect_point_states,
                                     default_reddit_specs,
                                     default_shopping_specs,
                                     destructive_place_order_specs,
                                     version_point_reach_specs)

    if args.site == "shopping":
        base = config.WA_SHOPPING
        if not (base and config.PRODUCT_URLS_PATH.exists()):
            print("ERROR: need WA_SHOPPING + product_urls.json (run `revact crawl`)")
            return 1
        products = json.loads(config.PRODUCT_URLS_PATH.read_text())
        specs = default_shopping_specs(
            products, base, n_cart=args.n_cart, n_wishlist=args.n_wishlist,
            n_compare=args.n_compare)
        if args.place_order:
            specs += destructive_place_order_specs(products, base, n=args.place_order)
    else:
        base = config.WA_REDDIT
        if not base:
            print("ERROR: need WA_REDDIT (source scripts/export_webarena_env.sh)")
            return 1
        default_submissions = [
            f"{base.rstrip('/')}/f/books/59421",
            f"{base.rstrip('/')}/f/books/81371",
            f"{base.rstrip('/')}/f/books/59447",
        ]
        default_forums = [
            f"{base.rstrip('/')}/f/books",
            f"{base.rstrip('/')}/f/AskReddit",
            f"{base.rstrip('/')}/f/technology",
        ]
        submissions = ([item.strip() for item in
                        args.reddit_submission_urls.split(",") if item.strip()]
                       or default_submissions)
        forums = ([item.strip() for item in args.reddit_forum_urls.split(",")
                   if item.strip()] or default_forums)
        specs = default_reddit_specs(
            submissions, forums, n_vote=args.n_reddit_vote,
            n_subscribe=args.n_reddit_subscribe)
    specs = version_point_reach_specs(specs, args.state_id_suffix)
    res = collect_point_states(
        specs, seed=args.seed, code_version=args.code_version)
    print(f"[collect-points] reached {res['n_reached']}/{len(specs)} "
          f"-> {res['state_bank']}")
    return 0 if res["n_reached"] else 1


def cmd_probe_points(args) -> int:
    """Execute reviewed point-level specs; persist only fully validated points."""
    from .grounding import save_formal_probe_results
    from .grounding.authoring import (ProbeAuthoringError,
                                      load_probe_execution_specs)
    from .grounding.point_runner import PointExecutionError, run_point_specs

    try:
        specs = load_probe_execution_specs(Path(args.spec))
    except (OSError, json.JSONDecodeError, ProbeAuthoringError, ValueError) as exc:
        print(f"ERROR: invalid execution spec: {exc}")
        return 1
    try:
        outcome = run_point_specs(specs, config.DATA_ROOT, commit=args.commit,
                                  headless=not args.no_headless)
    except PointExecutionError as exc:
        print(f"ERROR: point execution aborted: {exc}")
        return 1
    results, failures = outcome["results"], outcome["failures"]
    transitions = outcome.get("transitions") or []
    persisted: list[str] = []
    persist_error = ""
    if results:
        try:
            body, manifest = save_formal_probe_results(
                results, transitions=transitions)
            persisted = [ctx.probe_point_id for _r, ctx in results]
            print(f"formal points -> {body}\nformal manifest -> {manifest}")
        except Exception as exc:  # schema gate is intentionally fail closed
            persist_error = str(exc)
            print(f"ERROR: formal point persistence rejected: {exc}")
    report = {
        "spec": str(args.spec), "n_specs": len(specs),
        "n_executed": len(results), "n_persisted": len(persisted),
        "n_transitions_captured": len(transitions),
        "n_transitions_persisted": (
            len(transitions) if persisted and not persist_error else 0),
        "transition_body": str(
            config.DATA_ROOT / "grounded" / "transitions" /
            "probe_transitions.v1.jsonl"),
        "transition_manifest": str(
            config.DATA_ROOT / "grounded" / "transitions" /
            "TRANSITION_MANIFEST.v1.jsonl"),
        "persisted_point_ids": persisted, "failures": failures,
        "persist_error": persist_error, "commit": bool(args.commit),
    }
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
    for failure in failures:
        print(f"  [failed] {failure['probe_point_id']}: {failure['reason']}")
    if persist_error or not persisted:
        return 1
    return 0


def cmd_preflight_signal_observer(args) -> int:
    """Exercise registered DB observers without executing a web action.

    The resulting artifact proves read-only connectivity and query enforcement
    only.  It is deliberately marked ineligible for point-level evidence: one
    current-state read cannot be retrofitted onto a historical transition.
    """
    from datetime import datetime, timezone

    from .grounding.backend_observers import (
        BACKEND_PREFLIGHT_SCHEMA_VERSION, PROVIDERS,
        build_live_backend_observer)

    provider = PROVIDERS.get(args.site)
    if provider is None:
        print(f"ERROR: no backend observer provider for site {args.site!r}")
        return 1
    environment_instance = (
        args.environment_instance or config.site_base(args.site))
    if not environment_instance:
        print(f"ERROR: no environment instance for site {args.site!r}")
        return 1
    requested = [item.strip() for item in args.action_types.split(",")
                 if item.strip()]
    action_types = requested or sorted(provider.projections)
    unsupported = sorted(set(action_types) - set(provider.projections))
    if unsupported:
        print("ERROR: unsupported action types: " + ",".join(unsupported))
        return 1

    rows = []
    failures = []
    for action_type in action_types:
        try:
            observer = build_live_backend_observer(
                args.site, action_type, environment_instance,
                container=args.container or None)
            rows.append(observer.preflight())
        except Exception as exc:
            # BackendObserverError is already secret-safe; retaining the type
            # distinguishes schema/transport failures without stderr values.
            failures.append({
                "action_type": action_type,
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
    bundle = {
        "schema_version": BACKEND_PREFLIGHT_SCHEMA_VERSION + ".bundle.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(
            timespec="microseconds"),
        "code_version": args.code_version,
        "site": args.site,
        "environment_instance": environment_instance,
        "n_requested": len(action_types),
        "n_succeeded": len(rows),
        "n_failed": len(failures),
        "all_transactions_read_only": bool(rows) and all(
            row.get("transaction_read_only") is True for row in rows),
        "counts_as_point_signal_evidence": False,
        "results": rows,
        "failures": failures,
    }
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", encoding="utf-8") as handle:
            json.dump(bundle, handle, ensure_ascii=False, sort_keys=True,
                      indent=2, allow_nan=False)
            handle.write("\n")
    except FileExistsError:
        print(f"ERROR: refusing to overwrite immutable preflight {output}")
        return 1
    print(
        f"signal observer preflight {len(rows)}/{len(action_types)} "
        f"read-only -> {output}")
    return 0 if len(rows) == len(action_types) else 1


def cmd_prepare_point_batch(args) -> int:
    """Close state->candidate->reviewed execution-spec lineage offline."""
    from datetime import datetime, timezone

    from .grounding.batch_prepare import (PointBatchPreparationError,
                                           prepare_point_probe_batch)

    state_ids = [item.strip() for item in args.state_ids.split(",")
                 if item.strip()]
    output = (Path(args.output) if args.output else
              config.DATA_ROOT / "grounded" / "specs" /
              f"{args.probe_run_id}.jsonl")
    try:
        report = prepare_point_probe_batch(
            data_root=config.DATA_ROOT,
            state_ids=state_ids,
            probe_run_id=args.probe_run_id,
            reviewer=args.reviewer,
            timestamp=datetime.now(timezone.utc).isoformat(),
            code_version=args.code_version,
            execution_path=output,
            # Resolve each reviewed state's site from the central registry;
            # passing WA_SHOPPING here silently misrouted Reddit specs.
            environment_instance=None,
        )
    except (OSError, ValueError, PointBatchPreparationError) as exc:
        print(f"ERROR: point batch preparation failed: {exc}")
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def cmd_materialize_formal_candidates(args) -> int:
    """Freeze the point-scoped formal S4 body without touching legacy v3."""
    from .data.formal_candidates import (
        FormalCandidateMaterializationError,
        materialize_formal_candidates,
    )

    try:
        report = materialize_formal_candidates(
            Path(args.data_root),
            code_version=args.code_version,
            output_path=Path(args.output) if args.output else None,
            manifest_path=Path(args.manifest) if args.manifest else None,
            report_path=Path(args.report) if args.report else None,
        )
    except (FormalCandidateMaterializationError, OSError, ValueError) as exc:
        print(f"ERROR: formal candidate materialization failed: {exc}")
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def cmd_materialize_candidate_roles(args) -> int:
    """Freeze case-scoped role hypotheses and replayable evidence joins."""
    from .data.candidate_roles import (
        CandidateRoleError,
        materialize_candidate_roles,
    )

    try:
        report = materialize_candidate_roles(
            Path(args.data_root),
            code_version=args.code_version,
            output_path=Path(args.output) if args.output else None,
            manifest_path=Path(args.manifest) if args.manifest else None,
        )
    except (CandidateRoleError, OSError, ValueError) as exc:
        print(f"ERROR: candidate-role materialization failed: {exc}")
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def cmd_audit_mutation_census(args) -> int:
    """Verify the immutable live census; never executes a browser action."""
    from .data.mutation_miner import audit_live_mutation_census

    report = audit_live_mutation_census(Path(args.data_root))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["passed"] or args.allow_blocked else 1


def cmd_plan_mutation_census(args) -> int:
    """Create a read-only, alias-deduplicated review plan; executes nothing."""
    from .data.mutation_miner import (
        MutationMiningError,
        build_mutation_census_preflight,
        save_mutation_census_preflight,
    )

    try:
        report = build_mutation_census_preflight(
            Path(args.data_root), minimum_controls=args.minimum_controls,
            code_version=args.code_version)
        status = "not_written"
        if args.output:
            status = save_mutation_census_preflight(report, Path(args.output))
        report = {**report, "write_status": status,
                  "output": str(args.output or "")}
    except (MutationMiningError, OSError, ValueError, TypeError) as exc:
        print(f"ERROR: mutation census preflight failed: {exc}")
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["selection_reaches_minimum"] else 1


def cmd_capture_mutation_snapshots(args) -> int:
    """Capture full AX snapshots by registered-sandbox navigation only."""
    from .data.mutation_miner import (
        MutationMiningError,
        capture_full_mutation_snapshots,
    )

    targets = []
    for value in args.target:
        if "=" not in value:
            print("ERROR: each --target must be site=https://sandbox/path")
            return 1
        site, url = value.split("=", 1)
        targets.append((site, url))
    try:
        report = capture_full_mutation_snapshots(
            targets, body=Path(args.output), manifest=Path(args.manifest),
            code_version=args.code_version, headless=not args.no_headless)
    except (MutationMiningError, OSError, ValueError, TypeError) as exc:
        print(f"ERROR: full mutation snapshot capture failed: {exc}")
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


# --------------------------------------------------------------------------- #
# collect
# --------------------------------------------------------------------------- #
def cmd_collect(args) -> int:
    import os
    from .data.collect import run_collection
    from .policies import (IrisPolicy, LLMActionPolicy, ReadOnlyPolicyGuard,
                           ScriptedShoppingPolicy)

    os.environ["REVACT_WA_JUDGE"] = args.wa_judge
    if args.task_file:
        task_ids = json.loads(Path(args.task_file).read_text())
    elif args.task_ids:
        task_ids = [t.strip() for t in args.task_ids.split(",") if t.strip()]
    elif args.mock:
        task_ids = ["mock.add_to_cart", "mock.place_order", "mock.delete_address"]
    else:
        from .data.tasks import list_shopping_task_ids
        task_ids = list_shopping_task_ids()[: args.auto or 5]

    if args.mock:
        from .envs.mock_env import MockShoppingEnv

        def env_factory(tid):
            return MockShoppingEnv(goal=f"[mock:{tid.split('.')[-1]}] browse safely")

        def policy_factory(tid, seed):
            flow = tid.split(".")[-1]
            return ScriptedShoppingPolicy(
                flow if flow in ("add_to_cart", "place_order", "delete_address")
                else "add_to_cart")
    else:
        from .envs.harness import make_env

        def env_factory(tid):
            return make_env(tid, headless=not args.no_headless)

        def policy_factory(tid, seed):
            policy_cls = IrisPolicy if args.policy == "iris" else LLMActionPolicy
            policy = policy_cls(model=args.model or None,
                                base_url=args.base_url or None,
                                api_key_env=args.api_key_env,
                                provider=args.provider or None)
            return ReadOnlyPolicyGuard(policy) if args.read_only_live else policy

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    res = run_collection(env_factory, policy_factory, task_ids, seeds,
                         max_steps=args.max_steps, only_success=args.only_success,
                         save_screenshots=args.screenshots,
                         code_version=args.code_version)
    print(f"[done] success={res['n_success']}/{len(res['summaries'])}")
    return 0


def cmd_viz(args) -> int:
    from .viz import build_viz

    out = build_viz(out_path=Path(args.out) if args.out else None,
                    max_shots=args.max_shots,
                    full_document=not args.body_only)
    size = out.stat().st_size
    print(f"[viz] {out}  ({size / 1e6:.2f} MB, self-contained; open in a browser)")
    return 0


# --------------------------------------------------------------------------- #
# reach / crawl / scale
# --------------------------------------------------------------------------- #
def cmd_reach(args) -> int:
    from .data.reach import reach_and_record, save_reached

    base = config.WA_SHOPPING
    product = args.product_url or _default_product_url()
    if not (base and product):
        print("ERROR: need WA_SHOPPING + a product url")
        return 1
    renv = _make_live_renv(headless=not args.no_headless)
    try:
        records = reach_and_record(renv, base=base, product_url=product, seed=args.seed)
    finally:
        renv.close()
    path = save_reached(records)
    for r in records:
        print(f"  {r.name:15s} reached={r.reached} "
              f"risky={(r.risky_action or {}).get('text', '')[:50]!r}")
    print(f"reached {sum(r.reached for r in records)}/{len(records)} -> {path}")
    return 0


def cmd_crawl(args) -> int:
    from .envs.obs_utils import extract_interactive_bids

    base = config.WA_SHOPPING
    if not base:
        print("ERROR: WA_SHOPPING not set")
        return 1
    queries = ["phone", "cable", "book", "shoes", "cup", "light", "bag", "watch",
               "toy", "kitchen", "desk", "camera"]
    block = ["ascending", "descending", "direction", "next", "previous", "page",
             "compare", "wish list", "my cart", "sign out", "my account", "skip to",
             "advanced search", "store logo", "search", "home", "learn more"]

    def product_links(view):
        out = []
        for el in extract_interactive_bids(view.get("axtree_txt", "")):
            rest = el["line"].split("]", 1)[-1].strip()
            if not rest.startswith("link"):
                continue
            name = rest[len("link"):].strip().strip("'")
            if len(name) >= 25 and not any(b in name.lower() for b in block):
                out.append(el["bid"])
        return out

    renv = _make_live_renv()
    urls: list[str] = []
    try:
        renv.reset(seed=0)
        for q in queries:
            if len(urls) >= args.cap:
                break
            search = f"{base}/catalogsearch/result/?q={q}"
            for i in range(args.per_query):
                if len(urls) >= args.cap:
                    break
                _o, _r, _t, _tr, _i, v = renv.step(f"goto('{search}')")
                prods = product_links(v)
                if i >= len(prods):
                    break
                renv.step(f"click('{prods[i]}')")
                u = renv._last_obs_view.get("url", "")
                if u.endswith(".html") and "catalogsearch" not in u and u not in urls:
                    urls.append(u)
                    print(f"  [{len(urls)}] {u[:80]}")
    finally:
        renv.close()
    config.PRODUCT_URLS_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.PRODUCT_URLS_PATH.write_text(json.dumps(urls, indent=2))
    print(f"[crawl] {len(urls)} product urls -> {config.PRODUCT_URLS_PATH}")
    return 0


def cmd_scale(args) -> int:
    from .data.assemble import assemble
    from .data.scale import build_scaled_states, save_scaled

    base = config.WA_SHOPPING
    if not (base and config.PRODUCT_URLS_PATH.exists()):
        print("ERROR: need WA_SHOPPING + product_urls.json (run `revact crawl`)")
        return 1
    products = json.loads(config.PRODUCT_URLS_PATH.read_text())
    renv = _make_live_renv()
    try:
        records = build_scaled_states(renv, base, products,
                                      n_place_order=args.n_place_order)
    finally:
        renv.close()
    reached_path = save_scaled(records)
    print(f"[states] {len(records)} -> {reached_path}")
    res = assemble(reached_path, config.POINT_GROUNDING_PATH, config.DATA_ROOT)
    print(f"[assemble] SFT={res['n_sft']} DPO={res['n_dpo']}")
    if res.get("n_blocked_states"):
        print(f"  blocked states={res['n_blocked_states']} -> {res['report_path']}")
    return 0


# --------------------------------------------------------------------------- #
# assemble / split / train / eval / distill / inspect
# --------------------------------------------------------------------------- #
def cmd_assemble(args) -> int:
    from .data.assemble import assemble

    formal = not args.legacy_class_smoke
    if args.states:
        reached = Path(args.states)
    elif formal:
        from .grounding.batch_prepare import (
            PointBatchPreparationError, materialize_point_reached_states)
        try:
            state_report = materialize_point_reached_states(config.DATA_ROOT)
        except (OSError, ValueError, PointBatchPreparationError) as exc:
            print(f"ERROR: formal point-state materialization failed: {exc}")
            return 1
        reached = Path(state_report["output"])
        print(json.dumps(state_report, ensure_ascii=False, sort_keys=True))
    else:
        reached = config.STATE_BANK_DIR / "scaled_reached_states.jsonl"
        if not reached.exists():
            reached = config.STATE_BANK_DIR / "pilot_reached_states.jsonl"
    if not reached.exists():
        print(f"ERROR: no reached-states file under {config.STATE_BANK_DIR}")
        return 1
    grounding_path = (config.POINT_GROUNDING_PATH if formal else
                      config.REVERSIBILITY_PATH)
    if not grounding_path.exists():
        print(f"ERROR: missing {grounding_path} (run grounding migration/probes)")
        return 1
    res = assemble(reached, grounding_path, config.DATA_ROOT, formal=formal)
    print(f"[assemble] states={reached.name} SFT={res['n_sft']} DPO={res['n_dpo']}")
    print(f"  mode={'formal-point' if formal else 'LEGACY-QUARANTINE'} "
          f"blocked={res.get('n_blocked_states', 0)}")
    print(f"  SFT -> {res['sft_path']}\n  DPO -> {res['dpo_path']}")
    if formal and res["n_sft"] == 0:
        print("ERROR: formal assembly produced zero admitted samples; this is a "
              "blocked materialization, not a successful empty release.")
        return 1
    return 0


def cmd_import_grounding(args) -> int:
    """Admit externally measured point records through the canonical gate."""
    from .data.formal_import import FormalImportError, import_grounding_points

    try:
        report = import_grounding_points(Path(args.input), Path(args.data_root))
    except (FormalImportError, OSError, ValueError) as exc:
        print(f"ERROR: grounding import rejected: {exc}")
        return 1
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


def cmd_import_truth(args) -> int:
    """Admit explicitly authored normative truth; never infer it from recovery."""
    from .data.formal_import import FormalImportError, import_evaluation_truth

    try:
        report = import_evaluation_truth(Path(args.input), Path(args.data_root))
    except (FormalImportError, OSError, ValueError) as exc:
        print(f"ERROR: evaluation truth import rejected: {exc}")
        return 1
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


def cmd_author_truth(args) -> int:
    """Create explicit policy truth; recovery is forbidden as an input."""
    from .eval.truth_authoring import TruthAuthoringError, author_truth_file

    try:
        report = author_truth_file(
            Path(args.input), Path(args.output), data_root=Path(args.data_root))
    except (TruthAuthoringError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: policy truth authoring rejected: {exc}")
        return 1
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


def cmd_materialize_dpo(args) -> int:
    from .data.preferences import (PreferenceMaterializationError,
                                   materialize_formal_dpo)

    try:
        report = materialize_formal_dpo(
            Path(args.data_root), Path(args.negatives), family=args.family,
            trace_body=Path(args.trace_body) if args.trace_body else None,
            trace_manifest=(Path(args.trace_manifest)
                            if args.trace_manifest else None),
            output_path=Path(args.output) if args.output else None)
    except (PreferenceMaterializationError, OSError, ValueError) as exc:
        print(f"ERROR: formal DPO materialization rejected: {exc}")
        return 1
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


def cmd_author_dpo_negatives(args) -> int:
    """Freeze reviewed legal-candidate counterfactuals with honest provenance."""
    from .data.preferences import (
        PreferenceMaterializationError,
        author_legal_counterfactual_negatives,
    )

    try:
        report = author_legal_counterfactual_negatives(
            Path(args.data_root), Path(args.output), family=args.family,
            reviewer=args.reviewer, timestamp=args.timestamp)
    except (PreferenceMaterializationError, OSError, ValueError) as exc:
        print(f"ERROR: formal DPO negative authoring rejected: {exc}")
        return 1
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


def cmd_import_on_policy_traces(args) -> int:
    """Promote enriched rollout calls into immutable truth-verified traces."""
    from .data.governance import formal_release_context
    from .eval.on_policy import (OnPolicyTraceError, save_on_policy_traces,
                                 trace_from_rollout_episode)
    from .eval.truth import load_truth_records

    root = Path(args.data_root)
    try:
        episodes = [json.loads(line) for line in Path(args.rollout).open(
            encoding="utf-8") if line.strip()]
        context = formal_release_context(root)
        if context.grounding_error or context.truth_error:
            raise OnPolicyTraceError(
                "formal point/truth context is invalid: " +
                (context.grounding_error or context.truth_error))
        truths = load_truth_records(
            root / "eval" / "truth.jsonl", points=context.points)
        traces = []
        for episode in episodes:
            truth = truths.get(str(episode.get("evaluation_case_id") or ""))
            if truth is None:
                raise OnPolicyTraceError(
                    "rollout references unknown evaluation_case_id")
            steps = episode.get("steps") or []
            for step_index in range(len(steps)):
                traces.append(trace_from_rollout_episode(
                    episode, truth, step_index=step_index))
        body, manifest = save_on_policy_traces(
            traces, Path(args.output),
            Path(args.manifest) if args.manifest else None,
            truths=truths, append=args.append)
    except (OnPolicyTraceError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: on-policy trace import rejected: {exc}")
        return 1
    print(json.dumps({
        "n_traces": len(traces),
        "n_eligible_negatives": sum(trace.eligible_as_negative for trace in traces),
        "body": str(body), "manifest": str(manifest),
    }, ensure_ascii=False, sort_keys=True))
    return 0


def cmd_author_on_policy_traces(args) -> int:
    """Preflight a mixed rollout batch into valid and quarantined artifacts."""
    from .data.governance import formal_release_context
    from .eval.on_policy import (
        OnPolicyTraceError,
        author_on_policy_traces,
        save_on_policy_quarantine,
        save_on_policy_traces,
    )

    rollout = Path(args.rollout)
    root = Path(args.data_root)
    valid_body = Path(args.valid_body)
    valid_manifest = Path(args.valid_manifest)
    quarantine_body = Path(args.quarantine_body)
    quarantine_manifest = Path(args.quarantine_manifest)
    outputs = (
        valid_body, valid_manifest, quarantine_body, quarantine_manifest)
    try:
        if not rollout.is_file():
            raise OnPolicyTraceError(
                f"rollout JSONL does not exist or is not a file: {rollout}")
        resolved_outputs = [path.resolve() for path in outputs]
        if len(set(resolved_outputs)) != len(resolved_outputs):
            raise OnPolicyTraceError(
                "valid/quarantine body and manifest paths must be distinct")
        if valid_body.parent.resolve() != valid_manifest.parent.resolve():
            raise OnPolicyTraceError(
                "valid trace body and manifest must share one directory")
        if (quarantine_body.parent.resolve() !=
                quarantine_manifest.parent.resolve()):
            raise OnPolicyTraceError(
                "quarantine body and manifest must share one directory")
        if rollout.resolve() in resolved_outputs:
            raise OnPolicyTraceError(
                "rollout input cannot also be an output artifact")
        existing = [str(path) for path in outputs if path.exists()]
        if existing:
            raise OnPolicyTraceError(
                "refusing to overwrite on-policy authoring artifacts: "
                + ",".join(existing))

        episodes = []
        for line_no, line in enumerate(rollout.open(encoding="utf-8"), 1):
            if not line.strip():
                continue
            try:
                episodes.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise OnPolicyTraceError(
                    f"{rollout}:{line_no}: invalid JSON: {exc.msg}") from exc
        if not episodes:
            raise OnPolicyTraceError("rollout JSONL contains zero records")

        context = formal_release_context(root)
        if context.grounding_error or context.truth_error:
            raise OnPolicyTraceError(
                "formal point/truth context is invalid: "
                + (context.grounding_error or context.truth_error))
        if not context.truth:
            raise OnPolicyTraceError(
                "formal evaluation truth contains zero records")

        traces, rejections = author_on_policy_traces(episodes, context.truth)
        valid_paths: tuple[Path, Path] | None = None
        quarantine_paths: tuple[Path, Path] | None = None
        if traces:
            valid_paths = save_on_policy_traces(
                traces, valid_body, valid_manifest,
                truths=context.truth, append=False)
        if rejections:
            quarantine_paths = save_on_policy_quarantine(
                rejections, quarantine_body, quarantine_manifest)
    except (OnPolicyTraceError, OSError, ValueError,
            json.JSONDecodeError) as exc:
        print(f"ERROR: on-policy trace authoring rejected: {exc}")
        return 1

    print(json.dumps({
        "n_input_episodes": len(episodes),
        "n_valid": len(traces),
        "n_rejected": len(rejections),
        "n_eligible_negatives": sum(
            trace.eligible_as_negative for trace in traces),
        "valid_body": str(valid_paths[0]) if valid_paths else None,
        "valid_manifest": str(valid_paths[1]) if valid_paths else None,
        "quarantine_body": (
            str(quarantine_paths[0]) if quarantine_paths else None),
        "quarantine_manifest": (
            str(quarantine_paths[1]) if quarantine_paths else None),
    }, ensure_ascii=False, sort_keys=True))
    return 0


def cmd_import_episode_traces(args) -> int:
    """Freeze exact continuous stateless calls as canonical episode evidence."""
    from .data.episode_traces import (EpisodeTraceError,
                                      import_episode_trace_file)

    try:
        report = import_episode_trace_file(
            Path(args.input), Path(args.data_root), append=args.append)
    except (EpisodeTraceError, OSError, ValueError,
            json.JSONDecodeError) as exc:
        print(f"ERROR: stateless episode trace import rejected: {exc}")
        return 1
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


def cmd_preflight_episode_source(args) -> int:
    """Freeze a review-only source sheet; never author/import an episode."""
    from .data.episode_authoring import (
        EpisodeSourcePreflightError, preflight_episode_source,
    )

    try:
        report = preflight_episode_source(
            Path(args.run_manifest),
            args.trajectory_id,
            Path(args.output),
            Path(args.data_root),
        )
    except (EpisodeSourcePreflightError, OSError, ValueError,
            json.JSONDecodeError) as exc:
        print(f"ERROR: raw episode source preflight rejected: {exc}")
        return 1
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


def cmd_collect_opinions(args) -> int:
    """Collect an opinion baseline from label-blind pre-action evidence."""
    import os

    from .data.opinion_collect import (
        OpenAICompatibleOpinionClient,
        OpinionCollectionError,
        assert_opinion_input_manifest_integrity,
        build_opinion_messages,
        collect_opinion_records,
        validate_collection_settings,
    )
    from .data.opinions import save_opinion_records
    from .eval.truth import (assert_truth_manifest_integrity,
                             load_truth_records)
    from .grounding.schema import assert_manifest_integrity, load_probe_points

    root = Path(args.data_root)
    points_path = root / "grounded" / "probe_points.jsonl"
    point_manifest = root / "grounded" / "POINT_MANIFEST.jsonl"
    provider = args.provider or os.environ.get("REVACT_OPINION_PROVIDER", "")
    base_url = args.base_url or os.environ.get("REVACT_OPINION_BASE_URL", "")
    model = args.model or os.environ.get("REVACT_OPINION_MODEL", "")
    api_key_env = (
        args.api_key_env
        or os.environ.get("REVACT_OPINION_KEY_ENV", "")
        or "REVACT_OPINION_API_KEY"
    )
    try:
        temperature = (
            args.temperature
            if args.temperature is not None
            else float(os.environ.get("REVACT_OPINION_TEMPERATURE", "0"))
        )
        top_p = (
            args.top_p
            if args.top_p is not None
            else float(os.environ.get("REVACT_OPINION_TOP_P", "1"))
        )
        max_tokens = (
            args.max_tokens
            if args.max_tokens is not None
            else int(os.environ.get("REVACT_OPINION_MAX_TOKENS", "300"))
        )
        assert_manifest_integrity(points_path, point_manifest)
        points = load_probe_points(points_path, validate=True)
        truth_path = root / "eval" / "truth.jsonl"
        truth_manifest = root / "eval" / "TRUTH_MANIFEST.jsonl"
        assert_truth_manifest_integrity(truth_path, truth_manifest, points)
        truths = load_truth_records(truth_path, points=points)
        input_path = (Path(args.input) if args.input else
                      root / "opinions" / "inputs" /
                      "formal_request_constraint_inputs.v2.jsonl")
        input_manifest = (
            Path(args.input_manifest)
            if args.input_manifest
            else (root / "opinions" / "inputs" /
                  "FORMAL_REQUEST_CONSTRAINT_INPUT_MANIFEST.v2.jsonl"
                  if not args.input
                  else input_path.with_name(input_path.stem + ".manifest.jsonl"))
        )
        all_inputs = assert_opinion_input_manifest_integrity(
            input_path, input_manifest, points=points, truths=truths)
        if args.limit < 0:
            raise OpinionCollectionError("--limit must be non-negative")
        inputs = all_inputs[:args.limit] if args.limit else all_inputs
        if not inputs:
            raise OpinionCollectionError("opinion collection selected zero inputs")
        decode = validate_collection_settings(
            provider=provider,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=args.seed,
        )
        client = OpenAICompatibleOpinionClient(
            base_url=base_url,
            api_key_env=api_key_env,
            timeout=args.timeout,
        )
        if args.dry_run:
            messages = [build_opinion_messages(item) for item in inputs]
            print(json.dumps({
                "dry_run": True,
                "n_inputs": len(inputs),
                "n_eligible_input_cases": len(all_inputs),
                "eligible_evaluation_case_ids": [
                    item.evaluation_case_id for item in all_inputs],
                "selected_evaluation_case_ids": [
                    item.evaluation_case_id for item in inputs],
                "n_deferred_by_limit": len(all_inputs) - len(inputs),
                "provider": provider,
                "model": model,
                "base_url": client.base_url,
                "decode": decode,
                "model_facing_fields": ["goal", "pre_observation", "action"],
                "message_roles": [[message[0]["role"], message[1]["role"]]
                                  for message in messages],
                "credential_value_stored": False,
            }, ensure_ascii=False, sort_keys=True))
            return 0
        records, report = collect_opinion_records(
            inputs,
            client=client,
            provider=provider,
            model=model,
            rater_id=args.rater_id,
            collection_timestamp=args.timestamp,
            import_batch_id=args.batch_id,
            code_version=args.code_version,
            provenance_root=root,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=args.seed,
            instrument_id=args.instrument_id,
            instrument_version=args.instrument_version,
            points=points,
            truths=truths,
        )
        if args.output:
            output = Path(args.output)
        elif args.limit:
            output = (Path(args.data_root) / "opinions" /
                      f"opinion_labels.smoke_limit{args.limit}.v2.jsonl")
        else:
            output = Path(args.data_root) / "opinions" / "opinion_labels.v2.jsonl"
        manifest = (
            Path(args.manifest)
            if args.manifest
            else ((Path(args.data_root) / "opinions" /
                   "OPINION_LABEL_MANIFEST.v2.jsonl")
                  if not args.output and not args.limit
                  else output.with_name(output.stem + ".manifest.jsonl"))
        )
        save_opinion_records(
            records,
            output,
            manifest,
            append=args.append,
            points=points,
            truths=truths,
        )
    except (OpinionCollectionError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: opinion collection rejected: {exc}")
        return 1
    print(json.dumps({
        **report,
        "body": str(output),
        "manifest": str(manifest),
        "credential_value_stored": False,
    }, ensure_ascii=False, sort_keys=True))
    return 0


def cmd_prepare_opinion_inputs(args) -> int:
    """Freeze label-blind inputs for every eligible formal goal variant."""
    from .data.opinion_collect import (
        OpinionCollectionError,
        prepare_formal_opinion_inputs,
        save_opinion_rating_inputs,
    )

    try:
        variants = [item.strip() for item in args.variants.split(",")
                    if item.strip()]
        inputs, report = prepare_formal_opinion_inputs(
            Path(args.data_root), variants=variants, limit=args.limit)
        if args.output:
            output = Path(args.output)
        elif args.limit:
            output = (Path(args.data_root) / "opinions" / "inputs" /
                      f"formal_smoke_limit{args.limit}.v2.jsonl")
        else:
            output = Path(args.data_root) / "opinions" / "inputs" / \
                "formal_request_constraint_inputs.v2.jsonl"
        if args.manifest:
            manifest = Path(args.manifest)
        elif not args.output and not args.limit:
            manifest = (Path(args.data_root) / "opinions" / "inputs" /
                        "FORMAL_REQUEST_CONSTRAINT_INPUT_MANIFEST.v2.jsonl")
        else:
            manifest = output.with_name(output.stem + ".manifest.jsonl")
        save_opinion_rating_inputs(inputs, output, manifest)
    except (OpinionCollectionError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: opinion input preparation rejected: {exc}")
        return 1
    print(json.dumps({
        **report,
        "body": str(output),
        "manifest": str(manifest),
    }, ensure_ascii=False, sort_keys=True))
    return 0


def cmd_author_on_policy_negatives(args) -> int:
    from .data.preferences import (PreferenceMaterializationError,
                                   author_on_policy_negatives)

    try:
        report = author_on_policy_negatives(
            Path(args.data_root), Path(args.trace_body), Path(args.output),
            family=args.family, reviewer=args.reviewer,
            timestamp=args.timestamp,
            trace_manifest=(Path(args.trace_manifest)
                            if args.trace_manifest else None))
    except (PreferenceMaterializationError, OSError, ValueError) as exc:
        print(f"ERROR: on-policy negative authoring rejected: {exc}")
        return 1
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


def cmd_export_dataset(args) -> int:
    """Run the same release gates as the workbench, optionally without writes."""
    from .server.datasets import DataStore
    from .server.export import export_dataset

    report = export_dataset(DataStore(Path(args.data_root)), {
        "name": args.name, "formal": not args.legacy_development,
        "prefer_distilled": not args.no_distilled,
        "include_needs_review": args.include_needs_review,
        "dry_run": args.dry_run,
    })
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report.get("ok") else 1


def cmd_assemble_multiturn(args) -> int:
    from .data.multiturn import run

    res = run(formal=not args.legacy_class_smoke)
    print(f"[assemble-multiturn] SFT={res['n_sft']} DPO={res['n_dpo']} "
          f"skipped={res['n_skipped']}")
    if res.get("excluded"):
        print(f"  formal exclusions: {res['excluded']}")
    for s in res["skipped"]:
        print(f"  [skip] {s}")
    print(f"  SFT -> {res['sft_path']}\n  DPO -> {res['dpo_path']}")
    if not args.legacy_class_smoke and res["n_sft"] == 0:
        print("ERROR: formal multi-turn assembly admitted zero samples.")
        return 1
    return 0


def cmd_pin_dpo_supplement(args) -> int:
    """Create an immutable release manifest for one supplemental DPO body."""
    from .data.splits import write_formal_dpo_supplement_manifest

    body = Path(args.body)
    try:
        manifest = write_formal_dpo_supplement_manifest(
            body, release_id=args.release_id)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: DPO supplement pin rejected: {exc}")
        return 1
    print(json.dumps({
        "body": str(body),
        "manifest": str(manifest),
        "release_id": args.release_id,
    }, ensure_ascii=False, sort_keys=True))
    return 0


def cmd_split(args) -> int:
    from .data.splits import build_splits

    legacy = bool(args.legacy_development)
    if legacy:
        sft_path, dpo_path, out_dir = (
            config.SFT_PATH, config.DPO_PATH, config.SPLITS_DIR)
        mt_sft, mt_dpo = None, None
    else:
        sft_path, dpo_path, out_dir = (
            config.FORMAL_SFT_PATH, config.FORMAL_DPO_PATH,
            config.FORMAL_SPLITS_DIR)
        mt_sft, mt_dpo = (config.FORMAL_MULTITURN_SFT_PATH,
                          config.FORMAL_MULTITURN_DPO_PATH)
    additional_dpo = (() if legacy else tuple(sorted(
        config.FORMAL_TRAIN_DIR.glob(config.FORMAL_DPO_SUPPLEMENT_GLOB))))
    rep = build_splits(
        sft_path, dpo_path, out_dir, holdout_frac=args.holdout_frac,
        multiturn_sft_path=mt_sft, multiturn_dpo_path=mt_dpo,
        additional_dpo_paths=additional_dpo)
    mode = "LEGACY-DEVELOPMENT" if legacy else "formal-point"
    print(f"[split] mode={mode} source={sft_path} -> {out_dir}")
    print(f"held-out products ({len(rep['test_slugs'])}): {rep['test_slugs']}")
    print(f"SFT train={rep['n_train']} dev={rep.get('n_dev', 0)} "
          f"test={rep['n_test']} | "
          f"DPO train={rep['n_dpo_train']}")
    for split, dist in rep["distribution"].items():
        print(f"  {split}: {dist}")
    if not legacy and (rep["n_train"] == 0 or rep.get("n_dev", 0) == 0 or
                       rep["n_test"] == 0):
        print("ERROR: formal split requires non-empty train/dev/test partitions; "
              "the current corpus is blocked or lacks independent groups.")
        return 1
    return 0


def cmd_train(args) -> int:
    from .train.sft import run
    return run(train_path=Path(args.data) if args.data else None,
               dry_run=args.dry_run, output_dir=args.output_dir or None,
               max_len=args.max_len or None,
               allow_legacy=args.allow_legacy)


def cmd_train_dpo(args) -> int:
    from .train.dpo import run
    return run(train_path=Path(args.data) if args.data else None,
               dry_run=args.dry_run, output_dir=args.output_dir or None,
               beta=args.beta, max_steps=args.max_steps, adapter=args.adapter,
               allow_legacy=args.allow_legacy)


def cmd_train_grpo(args) -> int:
    from .train.grpo import run
    paths = [Path(p) for p in args.data.split(",") if p.strip()] if args.data else None
    return run(train_paths=paths, dry_run=args.dry_run,
               output_dir=args.output_dir or None, gspo=args.gspo,
               num_generations=args.num_generations, max_steps=args.max_steps,
               adapter=args.adapter, allow_legacy=args.allow_legacy)


def cmd_eval(args) -> int:
    from .eval.decisions import run
    return run(test_path=Path(args.data) if args.data else None,
               adapter=args.adapter, dry_run=args.dry_run,
               max_new_tokens=args.max_new_tokens,
               allow_legacy=args.legacy_development)


def cmd_distill(args) -> int:
    from .train.distill import run
    return run(
        in_path=Path(args.input) if args.input else None,
        out_path=Path(args.output) if args.output else None,
        limit=args.limit, family=args.family, overwrite=args.overwrite)


def cmd_eval_rollout(args) -> int:
    from .eval.rollout import (FormalRolloutError, load_formal_eval_cases,
                               run)
    from .policies import IrisPolicy, LLMActionPolicy

    site = "shopping"
    if args.exact_source_prompt_capture and (
            not args.code_version.strip() or args.budget != 1):
        print("ERROR: --exact-source-prompt-capture requires --budget 1 and "
              "a non-empty immutable --code-version")
        return 1
    if not args.legacy_development:
        try:
            # Validate point/truth/prompt joins before opening a browser or
            # constructing a network-backed policy.
            cases = load_formal_eval_cases(
                args.states, args.limit, data_root=Path(args.data_root),
                data_path=Path(args.data) if args.data else None)
            requested_variants = {
                variant for variant in args.variants.split(",")
                if variant.strip()
            }
            cases = [case for case in cases
                     if case.truth.variant in requested_variants]
            if args.site:
                cases = [case for case in cases
                         if case.point.site == args.site]
            if not cases:
                raise FormalRolloutError(
                    "no cases remain after variant/site selection")
            sites = {case.point.site for case in cases}
            if len(sites) != 1:
                raise FormalRolloutError(
                    "one live rollout invocation must contain exactly one site; "
                    f"got {sorted(sites)} (split invocations by site)")
            site = next(iter(sites))
        except (FormalRolloutError, ValueError, OSError) as exc:
            print(f"ERROR: formal rollout preflight failed: {exc}")
            return 1
    if not config.site_base(site):
        print(f"ERROR: site base is not configured for {site!r} "
              "(source scripts/export_webarena_env.sh)")
        return 1
    cls = IrisPolicy if args.policy == "iris" else LLMActionPolicy
    policy = cls(model=args.model or None, base_url=args.base_url or None,
                 api_key_env=args.api_key_env, provider=args.provider or None,
                 max_tokens=args.max_tokens)
    renv = _make_live_renv(headless=not args.no_headless, site=site)
    try:
        try:
            run(policy, tag=args.tag, which=args.states, limit=args.limit,
                budget=args.budget,
                variants=tuple(v for v in args.variants.split(",") if v.strip()),
                renv=renv, legacy_development=args.legacy_development,
                data_root=Path(args.data_root),
                data_path=Path(args.data) if args.data else None,
                site=args.site or None,
                exact_source_prompt_capture=args.exact_source_prompt_capture,
                code_version=args.code_version)
        except (FormalRolloutError, ValueError, OSError) as exc:
            print(f"ERROR: rollout evaluation failed: {exc}")
            return 1
    finally:
        renv.close()
    return 0


def cmd_eval_audit(args) -> int:
    from .eval.audit import EvaluationAuditError, run_audit

    try:
        report = run_audit(
            Path(args.input),
            output_path=Path(args.output) if args.output else None,
            formal=not args.legacy_development,
            bootstrap_iterations=args.bootstrap_iterations,
            overwrite=args.overwrite,
            data_root=Path(args.data_root))
    except (EvaluationAuditError, ValueError, OSError) as exc:
        print(f"ERROR: evaluation audit failed: {exc}")
        return 1
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


def cmd_serve(args) -> int:
    from .server.app import serve
    return serve(host=args.host, port=args.port)


def cmd_inspect(args) -> int:
    meta_path = config.RAW_DIR / "trajectories_meta.jsonl"
    if not meta_path.exists():
        print(f"no {meta_path}")
        return 1
    rows = [json.loads(ln) for ln in meta_path.open()]
    n_ok = sum(r.get("success", False) for r in rows)
    steps = [r.get("n_steps", 0) for r in rows]
    print(f"trajectories={len(rows)} success={n_ok} "
          f"({n_ok / max(len(rows), 1):.0%}) avg_steps={sum(steps)/max(len(steps),1):.1f}")
    ks_path = config.STATE_BANK_DIR / f"{config.SITE}_key_states.jsonl"
    if ks_path.exists():
        ks = [json.loads(ln) for ln in ks_path.open()]
        import collections
        types = collections.Counter(t for k in ks for t in k["afforded_action_types"])
        print(f"key states={len(ks)} by type={dict(types)}")
    return 0


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="revact", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("probe", help="run grounding probes")
    s.add_argument("names", nargs="*", help="probe names (see --list)")
    s.add_argument("--list", action="store_true")
    s.add_argument("--site", default="",
                   help="restrict to one site (shopping|shopping_admin|reddit)")
    s.add_argument("--all-nondestructive", action="store_true")
    s.add_argument("--commit", action="store_true",
                   help="destructive step (ALSO needs REVACT_ALLOW_DESTRUCTIVE=1)")
    s.add_argument("--mock", action="store_true")
    s.add_argument(
        "--formal-spec", default="",
        help="label-free JSON/JSONL point specs; writes canonical point body+manifest "
             "or fails without falling back to class-level labels")
    s.add_argument("--product-url", default="")
    s.add_argument("--submission-url", default="",
                   help="reddit: a submission page to probe (default books/59421)")
    s.add_argument("--forum-url", default="",
                   help="reddit: a forum page to probe (default /f/books)")
    s.add_argument("--budget", type=int, default=12)
    s.add_argument("--screenshots", action="store_true",
                   help="save a PNG per step under data/raw/screenshots/ (for `viz`)")
    s.add_argument("--no-headless", action="store_true")
    s.set_defaults(fn=cmd_probe)

    s = sub.add_parser(
        "collect-points",
        help="deterministically reach reproducible decision states with lineage")
    s.add_argument("--site", choices=("shopping", "reddit"), default="shopping")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--n-cart", type=int, default=10)
    s.add_argument("--n-wishlist", type=int, default=8)
    s.add_argument("--n-compare", type=int, default=8)
    s.add_argument("--place-order", type=int, default=0,
                   help="also reach N destructive place-order checkout states")
    s.add_argument("--n-reddit-vote", type=int, default=3)
    s.add_argument("--n-reddit-subscribe", type=int, default=3)
    s.add_argument(
        "--state-id-suffix", default="",
        help=("immutable recollection revision appended as '__<suffix>'; "
              "use when the logical state ID already exists"))
    s.add_argument("--reddit-submission-urls", default="",
                   help="comma-separated stable Postmill submission URLs")
    s.add_argument("--reddit-forum-urls", default="",
                   help="comma-separated stable Postmill forum URLs")
    s.add_argument(
        "--code-version", required=True,
        help="immutable commit/worktree identifier recorded in run/meta/state rows")
    s.set_defaults(fn=cmd_collect_points)

    s = sub.add_parser(
        "probe-points",
        help="execute reviewed point-level probe specs against the live site")
    s.add_argument("--spec", required=True,
                   help="iris.probe_execution_spec.v1 JSON/JSONL "
                        "(fixture PASSED + review APPROVED)")
    s.add_argument("--commit", action="store_true",
                   help="destructive step (ALSO needs REVACT_ALLOW_DESTRUCTIVE=1)")
    s.add_argument("--report", default="",
                   help="write the per-spec execution report JSON here")
    s.add_argument("--no-headless", action="store_true")
    s.set_defaults(fn=cmd_probe_points)

    s = sub.add_parser(
        "preflight-signal-observer",
        help="read-only live DB observer connectivity; never point evidence")
    s.add_argument("--site", choices=("shopping", "reddit"), required=True)
    s.add_argument(
        "--action-types", default="",
        help="comma-separated registered actions; empty runs every provider query")
    s.add_argument(
        "--environment-instance", default="",
        help="reviewed site base; defaults to the corresponding WA_* value")
    s.add_argument(
        "--container", default="",
        help="local reviewed Docker container (default from provider registry)")
    s.add_argument("--code-version", required=True)
    s.add_argument("--report", required=True,
                   help="new immutable JSON report; existing paths are refused")
    s.set_defaults(fn=cmd_preflight_signal_observer)

    s = sub.add_parser(
        "prepare-point-batch",
        help="materialize legal candidates and reviewed label-free point specs")
    s.add_argument("--state-ids", required=True,
                   help="comma-separated unique collected state_ids")
    s.add_argument("--probe-run-id", required=True)
    s.add_argument("--reviewer", required=True)
    s.add_argument("--code-version", required=True,
                   help="immutable commit/worktree identifier")
    s.add_argument("--output", default="",
                   help="immutable execution spec JSONL path")
    s.set_defaults(fn=cmd_prepare_point_batch)

    s = sub.add_parser(
        "materialize-formal-candidates",
        help="freeze deterministic 4--6 action sets for canonical point states")
    s.add_argument("--data-root", default=str(config.DATA_ROOT))
    s.add_argument("--code-version", required=True,
                   help="immutable commit/worktree identifier")
    s.add_argument("--output", default="",
                   help="optional formal candidate body path")
    s.add_argument("--manifest", default="",
                   help="optional formal candidate manifest path")
    s.add_argument("--report", default="",
                   help="optional immutable JSON materialization report")
    s.set_defaults(fn=cmd_materialize_formal_candidates)

    s = sub.add_parser(
        "materialize-candidate-roles",
        help="freeze case-scoped candidate-role proposals/evidence")
    s.add_argument("--data-root", default=str(config.DATA_ROOT))
    s.add_argument("--code-version", required=True,
                   help="immutable commit/worktree identifier")
    s.add_argument("--output", default="",
                   help="optional candidate-role body path")
    s.add_argument("--manifest", default="",
                   help="optional candidate-role manifest path")
    s.set_defaults(fn=cmd_materialize_candidate_roles)

    s = sub.add_parser(
        "audit-mutation-census",
        help="read-only verification of the release-scoped live control census")
    s.add_argument("--data-root", default=str(config.DATA_ROOT))
    s.add_argument(
        "--allow-blocked", action="store_true",
        help="return zero for an honest blocked report; never changes passed")
    s.set_defaults(fn=cmd_audit_mutation_census)

    s = sub.add_parser(
        "plan-mutation-census",
        help="inventory complete physical snapshots for human-reviewed live census")
    s.add_argument("--data-root", default=str(config.DATA_ROOT))
    s.add_argument("--minimum-controls", type=int, default=200)
    s.add_argument("--code-version", required=True,
                   help="immutable commit/worktree identifier")
    s.add_argument("--output", default="",
                   help="optional immutable preflight JSON path")
    s.set_defaults(fn=cmd_plan_mutation_census)

    s = sub.add_parser(
        "capture-mutation-snapshots",
        help="capture unpruned AX snapshots via sandbox navigation only")
    s.add_argument(
        "--target", action="append", required=True,
        help="registered site and URL as site=https://sandbox/path; repeatable")
    s.add_argument("--code-version", required=True,
                   help="immutable commit/worktree identifier")
    s.add_argument("--output", required=True,
                   help="immutable full-snapshot JSONL body")
    s.add_argument("--manifest", required=True,
                   help="immutable 1:1 full-snapshot manifest JSONL")
    s.add_argument("--no-headless", action="store_true")
    s.set_defaults(fn=cmd_capture_mutation_snapshots)

    s = sub.add_parser("collect", help="collect trajectories + key states")
    s.add_argument("--task-ids", default="")
    s.add_argument("--task-file", default="")
    s.add_argument("--auto", type=int, default=0)
    s.add_argument("--seeds", default="0")
    s.add_argument("--max-steps", type=int, default=25)
    s.add_argument(
        "--code-version", default="",
        help="immutable commit/worktree identifier stamped into new raw "
             "episode-source steps and collection manifest")
    s.add_argument("--only-success", action="store_true")
    s.add_argument("--mock", action="store_true")
    s.add_argument("--policy", choices=("collector", "iris"),
                   default="collector",
                   help="iris captures deployment-shaped full IRIS completions; "
                        "collector retains the plain-action discovery policy")
    s.add_argument("--model", default="")
    s.add_argument("--base-url", default="",
                   help="OpenAI-compatible policy endpoint")
    s.add_argument("--provider", default="",
                   help="provider provenance label; inferred from base URL if empty")
    s.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    s.add_argument(
        "--read-only-live", action="store_true",
        help="fail closed before ambiguous or state-changing live actions")
    s.add_argument(
        "--wa-judge",
        choices=["off", "route", "openrouter", "deepseek", "openai"],
        default="off",
        help="off, native OpenAI, or a configured OpenAI-compatible route")
    s.add_argument("--screenshots", action="store_true",
                   help="save a PNG per step under data/raw/screenshots/ (for `viz`)")
    s.add_argument("--no-headless", action="store_true")
    s.set_defaults(fn=cmd_collect)

    s = sub.add_parser("reach", help="reach pilot risk states")
    s.add_argument("--product-url", default="")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--no-headless", action="store_true")
    s.set_defaults(fn=cmd_reach)

    s = sub.add_parser("crawl", help="harvest product urls")
    s.add_argument("--cap", type=int, default=40)
    s.add_argument("--per-query", type=int, default=6)
    s.set_defaults(fn=cmd_crawl)

    s = sub.add_parser("scale", help="build scaled states + assemble")
    s.add_argument("--n-place-order", type=int, default=6)
    s.set_defaults(fn=cmd_scale)

    s = sub.add_parser("assemble", help="grounded states + labels -> SFT/DPO")
    s.add_argument("--states", default="")
    s.add_argument(
        "--legacy-class-smoke", action="store_true",
        help="development only: class-level labels -> quarantined legacy outputs")
    s.set_defaults(fn=cmd_assemble)

    s = sub.add_parser(
        "import-grounding",
        help="validate and append measured point-level grounding JSONL")
    s.add_argument("--input", required=True,
                   help="immutable JSONL of complete iris.grounding.point.v1 rows")
    s.add_argument("--data-root", default=str(config.DATA_ROOT))
    s.set_defaults(fn=cmd_import_grounding)

    s = sub.add_parser(
        "import-eval-truth",
        help="validate and append explicitly authored point×variant policy truth")
    s.add_argument("--input", required=True,
                   help="JSONL of complete iris.evaluation.truth.v1 rows")
    s.add_argument("--data-root", default=str(config.DATA_ROOT))
    s.set_defaults(fn=cmd_import_truth)

    s = sub.add_parser(
        "author-truth",
        help="create point×variant truth from explicit normative policy input")
    s.add_argument("--input", required=True,
                   help="iris.policy_truth_authoring.v1 JSON")
    s.add_argument("--output", required=True,
                   help="immutable authored truth JSONL (not canonical body)")
    s.add_argument("--data-root", default=str(config.DATA_ROOT))
    s.set_defaults(fn=cmd_author_truth)

    s = sub.add_parser(
        "author-dpo-negatives",
        help="freeze legal-candidate counterfactuals with explicit synthetic prose provenance")
    s.add_argument("--family", choices=("single", "multiturn"), required=True)
    s.add_argument("--output", required=True)
    s.add_argument("--reviewer", required=True)
    s.add_argument("--timestamp", required=True)
    s.add_argument("--data-root", default=str(config.DATA_ROOT))
    s.set_defaults(fn=cmd_author_dpo_negatives)

    s = sub.add_parser(
        "import-on-policy-traces",
        help="verify enriched formal rollout calls and freeze immutable trace evidence")
    s.add_argument("--rollout", required=True)
    s.add_argument("--output", required=True,
                   help="versioned iris.on_policy_step.v1 JSONL body")
    s.add_argument("--manifest", default="",
                   help="optional manifest path (derived beside body by default)")
    s.add_argument("--data-root", default=str(config.DATA_ROOT))
    s.add_argument("--append", action="store_true")
    s.set_defaults(fn=cmd_import_on_policy_traces)

    s = sub.add_parser(
        "author-on-policy-traces",
        help="preflight every rollout step; save valid traces and minimal "
             "secret-free rejections separately",
        description="Validate every step against immutable point-level truth. "
                    "Valid traces and minimal secret-free rejection records "
                    "are written as separate body/manifest pairs. Existing "
                    "outputs are never overwritten.")
    s.add_argument("--rollout", required=True,
                   help="enriched formal rollout episode JSONL")
    s.add_argument("--data-root", default=str(config.DATA_ROOT),
                   help="formal point/truth body+manifest root")
    s.add_argument("--valid-body", required=True,
                   help="new iris.on_policy_step.v1 JSONL body")
    s.add_argument("--valid-manifest", required=True,
                   help="new 1:1 manifest for --valid-body")
    s.add_argument("--quarantine-body", required=True,
                   help="new minimal rejection JSONL (no prompts/completions)")
    s.add_argument("--quarantine-manifest", required=True,
                   help="new 1:1 manifest for --quarantine-body")
    s.set_defaults(fn=cmd_author_on_policy_traces)

    s = sub.add_parser(
        "import-episode-traces",
        help="validate exact continuous stateless policy calls and freeze "
             "the canonical episode body+manifest")
    s.add_argument("--input", required=True,
                   help="iris.stateless_episode_trace.v1 JSONL; legacy raw "
                        "StepRecord trajectories are rejected")
    s.add_argument("--data-root", default=str(config.DATA_ROOT))
    s.add_argument("--append", action="store_true")
    s.set_defaults(fn=cmd_import_episode_traces)

    s = sub.add_parser(
        "preflight-episode-source",
        help="strictly verify one raw collection trajectory and write an "
             "immutable review-only sheet (never a canonical episode)",
        description="Strictly verify one raw collection trajectory and write "
                    "an immutable review-only sheet (never a canonical "
                    "episode). It does not author or import one.")
    s.add_argument(
        "--run-manifest", required=True,
        help="completed iris.collection-run.v2 JSON under "
             "DATA_ROOT/manifests/collection_runs")
    s.add_argument(
        "--trajectory-id", required=True,
        help="physical trajectory_id including the __run_<run_id> suffix")
    s.add_argument(
        "--output", required=True,
        help="new immutable JSON review sheet; it has counts_as_formal=false "
             "and cannot be imported as a canonical episode")
    s.add_argument("--data-root", default=str(config.DATA_ROOT))
    s.set_defaults(fn=cmd_preflight_episode_source)

    s = sub.add_parser(
        "author-on-policy-negatives",
        help="derive DPO negatives only from verified raw model-error traces")
    s.add_argument("--family", choices=("single", "multiturn"), required=True)
    s.add_argument("--trace-body", required=True)
    s.add_argument("--trace-manifest", default="")
    s.add_argument("--output", required=True)
    s.add_argument("--reviewer", required=True)
    s.add_argument("--timestamp", required=True)
    s.add_argument("--data-root", default=str(config.DATA_ROOT))
    s.set_defaults(fn=cmd_author_on_policy_negatives)

    s = sub.add_parser(
        "prepare-opinion-inputs",
        help="freeze label-blind point×goal opinion input body+manifest")
    s.add_argument("--data-root", default=str(config.DATA_ROOT))
    s.add_argument("--variants", default="constraint,request",
                   help="comma-separated formal truth variants")
    s.add_argument("--limit", type=int, default=0,
                   help="deterministic case-id prefix; 0 selects all eligible")
    s.add_argument("--output", default="",
                   help="immutable body; smoke limits get a separate default path")
    s.add_argument("--manifest", default="")
    s.set_defaults(fn=cmd_prepare_opinion_inputs)

    s = sub.add_parser(
        "collect-opinions",
        help="collect label-blind pre-action LLM opinion ratings")
    s.add_argument("--input", default="",
                   help="strict v2 label-blind input JSONL")
    s.add_argument("--input-manifest", default="",
                   help="1:1 input manifest (derived beside custom input by default)")
    s.add_argument("--output", default="",
                   help="versioned body; smoke limits get a separate default path")
    s.add_argument("--manifest", default="",
                   help="canonical opinion manifest")
    s.add_argument("--data-root", default=str(config.DATA_ROOT))
    s.add_argument("--provider", default="",
                   help="provider provenance label; or REVACT_OPINION_PROVIDER")
    s.add_argument("--base-url", default="",
                   help="explicit OpenAI-compatible root; or REVACT_OPINION_BASE_URL")
    s.add_argument("--model", default="",
                   help="requested model; or REVACT_OPINION_MODEL")
    s.add_argument("--api-key-env", default="",
                   help="credential env name; value is never persisted")
    s.add_argument("--rater-id", required=True,
                   help="pseudonymous stable rater/config identity")
    s.add_argument("--timestamp", required=True,
                   help="collection timestamp with timezone")
    s.add_argument("--batch-id", required=True,
                   help="immutable opinion collection batch ID")
    s.add_argument("--code-version", required=True,
                   help="immutable commit/worktree snapshot ID")
    s.add_argument("--instrument-id", default="iris-preaction-opinion-rater")
    s.add_argument("--instrument-version", default="v1")
    s.add_argument("--temperature", type=float, default=None)
    s.add_argument("--top-p", type=float, default=None)
    s.add_argument("--max-tokens", type=int, default=None)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--timeout", type=int, default=90)
    s.add_argument("--limit", type=int, default=0,
                   help="deterministic input prefix for 2--4 case smoke; 0=all")
    s.add_argument("--append", action="store_true")
    s.add_argument("--dry-run", action="store_true",
                   help="validate exact joins/messages/config without key, calls, or writes")
    s.set_defaults(fn=cmd_collect_opinions)

    s = sub.add_parser(
        "materialize-dpo",
        help="join reviewed legal/on-policy errors to formal point SFT")
    s.add_argument("--negatives", required=True,
                   help="iris.preference_negative.v1/v2 JSONL")
    s.add_argument("--trace-body", default="",
                   help="required for trace-backed on_policy negatives")
    s.add_argument("--trace-manifest", default="")
    s.add_argument("--output", default="",
                   help="optional immutable output path; default active versioned artifact")
    s.add_argument("--family", choices=["single", "multiturn"],
                   default="single")
    s.add_argument("--data-root", default=str(config.DATA_ROOT))
    s.set_defaults(fn=cmd_materialize_dpo)

    s = sub.add_parser(
        "export-dataset",
        help="run dataset release gates and optionally materialize a release")
    s.add_argument("--name", default="release")
    s.add_argument("--data-root", default=str(config.DATA_ROOT))
    s.add_argument("--dry-run", action="store_true",
                   help="execute all gates and report would-write files only")
    s.add_argument("--legacy-development", action="store_true")
    s.add_argument("--no-distilled", action="store_true")
    s.add_argument("--include-needs-review", action="store_true")
    s.set_defaults(fn=cmd_export_dataset)

    s = sub.add_parser("assemble-multiturn",
                       help="trajectories + key states -> multi-turn SFT/DPO")
    s.add_argument(
        "--legacy-class-smoke", action="store_true",
        help="development only: class-level labels -> quarantined legacy outputs")
    s.set_defaults(fn=cmd_assemble_multiturn)

    s = sub.add_parser("split", help="product-level train/test split")
    s.add_argument("--holdout-frac", type=float, default=0.25)
    s.add_argument(
        "--legacy-development", action="store_true",
        help="explicitly split frozen pilot artifacts; never used by formal trainers")
    s.set_defaults(fn=cmd_split)

    s = sub.add_parser(
        "pin-dpo-supplement",
        help="immutably hash-pin a formal supplemental DPO body to a release")
    s.add_argument("--body", required=True,
                   help="completed formal supplemental DPO JSONL")
    s.add_argument("--release-id", required=True,
                   help="immutable publication/release identifier")
    s.set_defaults(fn=cmd_pin_dpo_supplement)

    s = sub.add_parser("train-dpo", help="LoRA DPO on preference pairs (TRL)")
    s.add_argument("--data", default="",
                   help="pairs jsonl (default formal/splits/dpo_train.jsonl)")
    s.add_argument("--adapter", default="",
                   help="SFT LoRA dir to warm-start from (merged; ref = SFT policy)")
    s.add_argument("--beta", type=float, default=0.1)
    s.add_argument("--max-steps", type=int, default=-1)
    s.add_argument("--output-dir", default="")
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--allow-legacy", action="store_true",
                   help="audit quarantined legacy pairs (requires --dry-run; never trains)")
    s.set_defaults(fn=cmd_train_dpo)

    s = sub.add_parser("train-grpo",
                       help="GRPO/GSPO with grounded verifiable rewards (TRL)")
    s.add_argument("--data", default="",
                   help="comma-separated SFT jsonl paths (default formal/splits/sft_train.jsonl)")
    s.add_argument("--gspo", action="store_true",
                   help="sequence-level importance sampling (GSPO)")
    s.add_argument("--adapter", default="",
                   help="SFT LoRA dir to warm-start the policy from (merged)")
    s.add_argument("--num-generations", type=int, default=4)
    s.add_argument("--max-steps", type=int, default=-1)
    s.add_argument("--output-dir", default="")
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--allow-legacy", action="store_true",
                   help="audit quarantined legacy prompts (requires --dry-run; never trains)")
    s.set_defaults(fn=cmd_train_grpo)

    s = sub.add_parser("train", help="LoRA SFT (single-step and/or multi-turn)")
    s.add_argument("--data", default="",
                   help="sft jsonl (default formal/splits/sft_train.jsonl)")
    s.add_argument("--max-len", type=int, default=0,
                   help="override train.max_len (multi-turn needs ~8192)")
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--allow-legacy", action="store_true",
                   help="audit quarantined legacy rows (requires --dry-run; never trains)")
    s.add_argument("--output-dir", default="")
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("eval", help="decision accuracy on held-out split")
    s.add_argument("--data", default="",
                   help="test JSONL (default: formal/splits/sft_test.jsonl)")
    s.add_argument("--adapter", default="")
    s.add_argument("--dry-run", action="store_true")
    s.add_argument(
        "--legacy-development", action="store_true",
        help="explicit frozen-pilot diagnostic; never reported as formal evidence")
    s.add_argument("--max-new-tokens", type=int, default=0,
                   help="0=auto gold-completion p99 + safety margin")
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("distill", help="teacher prose distillation (pinned labels)")
    s.add_argument("--limit", type=int, default=10)
    s.add_argument("--family", choices=("single", "multiturn", "all"),
                   default="all",
                   help="formal SFT family; limit applies independently per family")
    s.add_argument("--input", default="",
                   help="custom formal input JSONL (requires a single family)")
    s.add_argument("--output", default="",
                   help="custom distilled JSONL (requires a single family)")
    s.add_argument("--overwrite", action="store_true")
    s.set_defaults(fn=cmd_distill)

    s = sub.add_parser("eval-rollout",
                       help="point/truth-grounded live rollout metrics; guarded "
                            "destructive targets are never executed")
    s.add_argument("--policy", default="iris", choices=["iris", "collector"],
                   help="iris = trained format (<answer> field); collector = plain")
    s.add_argument("--model", default="", help="served model name")
    s.add_argument("--base-url", default="",
                   help="OpenAI-compatible endpoint, e.g. http://127.0.0.1:8000/v1")
    s.add_argument("--provider", default="",
                   help="provenance label (inferred from base URL when omitted)")
    s.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    s.add_argument("--tag", default="rollout")
    s.add_argument("--states", default="test", choices=["test", "all"],
                   help="test = held-out-product states only")
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--budget", type=int, default=3)
    s.add_argument("--variants", default="constraint,request")
    s.add_argument(
        "--site", choices=["shopping", "shopping_admin", "reddit"],
        default="",
        help="restrict formal cases to one concrete site before live preflight")
    s.add_argument("--max-tokens", type=int, default=512)
    s.add_argument("--no-headless", action="store_true")
    s.add_argument("--data-root", default=str(config.DATA_ROOT),
                   help="formal point/truth/prompt root")
    s.add_argument("--data", default="",
                   help="optional formal SFT evaluation rows; exact joins still required")
    s.add_argument(
        "--exact-source-prompt-capture", action="store_true",
        help="first call uses the byte-exact serialized SFT prompt; intended "
             "for trace-backed on-policy DPO capture, not free-running rollout")
    s.add_argument(
        "--code-version", required=False, default="",
        help="immutable commit/worktree snapshot id; required for verified "
             "on-policy trace authoring")
    s.add_argument(
        "--legacy-development", action="store_true",
        help="explicit frozen-pilot keyword/class proxy; FSR is never claimable")
    s.set_defaults(fn=cmd_eval_rollout)

    s = sub.add_parser(
        "eval-audit",
        help="recompute metrics/noise/bootstrap from saved point-level episodes")
    s.add_argument("--input", required=True, help="saved rollout/prediction JSONL")
    s.add_argument("--output", default="", help="optional immutable JSON report")
    s.add_argument("--bootstrap-iterations", type=int, default=2000)
    s.add_argument("--data-root", default=str(config.DATA_ROOT),
                   help="formal point/truth body+manifest root")
    s.add_argument("--overwrite", action="store_true")
    s.add_argument(
        "--legacy-development", action="store_true",
        help="accept historical rows but keep independent-truth FSR unclaimable")
    s.set_defaults(fn=cmd_eval_audit)

    s = sub.add_parser("inspect", help="dataset quality stats")
    s.set_defaults(fn=cmd_inspect)

    s = sub.add_parser("viz", help="build the self-contained dataset explorer HTML")
    s.add_argument("--out", default="", help="output path (default outputs/dataset_viz.html)")
    s.add_argument("--max-shots", type=int, default=60,
                   help="cap on embedded screenshot thumbnails")
    s.add_argument("--body-only", action="store_true",
                   help="emit content without the <html> skeleton (artifact publishing)")
    s.set_defaults(fn=cmd_viz)

    s = sub.add_parser("serve", help="dataset construction workbench (web UI)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=7788)
    s.set_defaults(fn=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
