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
                                     default_shopping_specs,
                                     destructive_place_order_specs)

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
    res = collect_point_states(specs, seed=args.seed)
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
    persisted: list[str] = []
    persist_error = ""
    if results:
        try:
            body, manifest = save_formal_probe_results(results)
            persisted = [ctx.probe_point_id for _r, ctx in results]
            print(f"formal points -> {body}\nformal manifest -> {manifest}")
        except Exception as exc:  # schema gate is intentionally fail closed
            persist_error = str(exc)
            print(f"ERROR: formal point persistence rejected: {exc}")
    report = {
        "spec": str(args.spec), "n_specs": len(specs),
        "n_executed": len(results), "n_persisted": len(persisted),
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


# --------------------------------------------------------------------------- #
# collect
# --------------------------------------------------------------------------- #
def cmd_collect(args) -> int:
    import os
    from .data.collect import run_collection
    from .policies import LLMActionPolicy, ScriptedShoppingPolicy

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
            return LLMActionPolicy(model=args.model or None,
                                   api_key_env=args.api_key_env)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    res = run_collection(env_factory, policy_factory, task_ids, seeds,
                         max_steps=args.max_steps, only_success=args.only_success,
                         save_screenshots=args.screenshots)
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

    reached = Path(args.states) if args.states else \
        config.STATE_BANK_DIR / "scaled_reached_states.jsonl"
    if not reached.exists():
        reached = config.STATE_BANK_DIR / "pilot_reached_states.jsonl"
    if not reached.exists():
        print(f"ERROR: no reached-states file under {config.STATE_BANK_DIR}")
        return 1
    formal = not args.legacy_class_smoke
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


def cmd_materialize_dpo(args) -> int:
    from .data.preferences import (PreferenceMaterializationError,
                                   materialize_formal_dpo)

    try:
        report = materialize_formal_dpo(
            Path(args.data_root), Path(args.negatives), family=args.family)
    except (PreferenceMaterializationError, OSError, ValueError) as exc:
        print(f"ERROR: formal DPO materialization rejected: {exc}")
        return 1
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


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
    rep = build_splits(
        sft_path, dpo_path, out_dir, holdout_frac=args.holdout_frac,
        multiturn_sft_path=mt_sft, multiturn_dpo_path=mt_dpo)
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
    return run(limit=args.limit)


def cmd_eval_rollout(args) -> int:
    from .eval.rollout import (FormalRolloutError, load_formal_eval_cases,
                               run)
    from .policies import IrisPolicy, LLMActionPolicy

    site = "shopping"
    if not args.legacy_development:
        try:
            # Validate point/truth/prompt joins before opening a browser or
            # constructing a network-backed policy.
            cases = load_formal_eval_cases(
                args.states, args.limit, data_root=Path(args.data_root),
                data_path=Path(args.data) if args.data else None)
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
                 api_key_env=args.api_key_env, max_tokens=args.max_tokens)
    renv = _make_live_renv(headless=not args.no_headless, site=site)
    try:
        try:
            run(policy, tag=args.tag, which=args.states, limit=args.limit,
                budget=args.budget,
                variants=tuple(v for v in args.variants.split(",") if v.strip()),
                renv=renv, legacy_development=args.legacy_development,
                data_root=Path(args.data_root),
                data_path=Path(args.data) if args.data else None)
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
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--n-cart", type=int, default=10)
    s.add_argument("--n-wishlist", type=int, default=8)
    s.add_argument("--n-compare", type=int, default=8)
    s.add_argument("--place-order", type=int, default=0,
                   help="also reach N destructive place-order checkout states")
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

    s = sub.add_parser("collect", help="collect trajectories + key states")
    s.add_argument("--task-ids", default="")
    s.add_argument("--task-file", default="")
    s.add_argument("--auto", type=int, default=0)
    s.add_argument("--seeds", default="0")
    s.add_argument("--max-steps", type=int, default=25)
    s.add_argument("--only-success", action="store_true")
    s.add_argument("--mock", action="store_true")
    s.add_argument("--model", default="")
    s.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    s.add_argument("--wa-judge", choices=["off", "deepseek", "openai"], default="off")
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
        "materialize-dpo",
        help="join reviewed legal/on-policy errors to formal point SFT")
    s.add_argument("--negatives", required=True,
                   help="iris.preference_negative.v1 JSONL")
    s.add_argument("--family", choices=["single", "multiturn"],
                   default="single")
    s.add_argument("--data-root", default=str(config.DATA_ROOT))
    s.set_defaults(fn=cmd_materialize_dpo)

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
    s.set_defaults(fn=cmd_distill)

    s = sub.add_parser("eval-rollout",
                       help="point/truth-grounded live rollout metrics; guarded "
                            "destructive targets are never executed")
    s.add_argument("--policy", default="iris", choices=["iris", "collector"],
                   help="iris = trained format (<answer> field); collector = plain")
    s.add_argument("--model", default="", help="served model name")
    s.add_argument("--base-url", default="",
                   help="OpenAI-compatible endpoint, e.g. http://127.0.0.1:8000/v1")
    s.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    s.add_argument("--tag", default="rollout")
    s.add_argument("--states", default="test", choices=["test", "all"],
                   help="test = held-out-product states only")
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--budget", type=int, default=3)
    s.add_argument("--variants", default="constraint,request")
    s.add_argument("--max-tokens", type=int, default=512)
    s.add_argument("--no-headless", action="store_true")
    s.add_argument("--data-root", default=str(config.DATA_ROOT),
                   help="formal point/truth/prompt root")
    s.add_argument("--data", default="",
                   help="optional formal SFT evaluation rows; exact joins still required")
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
