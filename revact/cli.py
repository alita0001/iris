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
    from .grounding import list_probes, run_probe, save_results
    from .grounding import probes  # noqa: F401  (registers all probes)

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
                r = run_probe(name, ctx)
                if renv.save_screenshots:
                    r.evidence["screenshots_dir"] = str(
                        Path("raw") / "screenshots" / renv.trajectory_id)
                results.append(r)
                ev = {k: v for k, v in r.evidence.items() if k != "undo_actions"}
                print(f"  [{site}] {name:30s} -> {r.label:22s} {ev}")
        finally:
            renv.close()

    if not args.mock:
        path = save_results(results)
        print(f"saved -> {path}")
    return 0


def get_probe_site(name: str) -> str:
    from .grounding import get_probe
    return get_probe(name).site


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
    res = assemble(reached_path, config.REVERSIBILITY_PATH, config.DATA_ROOT)
    print(f"[assemble] SFT={res['n_sft']} DPO={res['n_dpo']}")
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
    if not config.REVERSIBILITY_PATH.exists():
        print(f"ERROR: missing {config.REVERSIBILITY_PATH} (run `revact probe`)")
        return 1
    res = assemble(reached, config.REVERSIBILITY_PATH, config.DATA_ROOT)
    print(f"[assemble] states={reached.name} SFT={res['n_sft']} DPO={res['n_dpo']}")
    print(f"  SFT -> {res['sft_path']}\n  DPO -> {res['dpo_path']}")
    return 0


def cmd_assemble_multiturn(args) -> int:
    from .data.multiturn import run

    res = run()
    print(f"[assemble-multiturn] SFT={res['n_sft']} DPO={res['n_dpo']} "
          f"skipped={res['n_skipped']}")
    for s in res["skipped"]:
        print(f"  [skip] {s}")
    print(f"  SFT -> {res['sft_path']}\n  DPO -> {res['dpo_path']}")
    return 0


def cmd_split(args) -> int:
    from .data.splits import build_splits

    rep = build_splits(config.SFT_PATH, config.DPO_PATH, config.SPLITS_DIR,
                       holdout_frac=args.holdout_frac)
    print(f"held-out products ({len(rep['test_slugs'])}): {rep['test_slugs']}")
    print(f"SFT train={rep['n_train']} test={rep['n_test']} | "
          f"DPO train={rep['n_dpo_train']}")
    for split, dist in rep["distribution"].items():
        print(f"  {split}: {dist}")
    return 0


def cmd_train(args) -> int:
    from .train.sft import run
    return run(train_path=Path(args.data) if args.data else None,
               dry_run=args.dry_run, output_dir=args.output_dir or None,
               max_len=args.max_len or None)


def cmd_train_dpo(args) -> int:
    from .train.dpo import run
    return run(train_path=Path(args.data) if args.data else None,
               dry_run=args.dry_run, output_dir=args.output_dir or None,
               beta=args.beta, max_steps=args.max_steps, adapter=args.adapter)


def cmd_train_grpo(args) -> int:
    from .train.grpo import run
    paths = [Path(p) for p in args.data.split(",") if p.strip()] if args.data else None
    return run(train_paths=paths, dry_run=args.dry_run,
               output_dir=args.output_dir or None, gspo=args.gspo,
               num_generations=args.num_generations, max_steps=args.max_steps,
               adapter=args.adapter)


def cmd_eval(args) -> int:
    from .eval.decisions import run
    return run(adapter=args.adapter, dry_run=args.dry_run,
               max_new_tokens=args.max_new_tokens)


def cmd_distill(args) -> int:
    from .train.distill import run
    return run(limit=args.limit)


def cmd_eval_rollout(args) -> int:
    from .eval.rollout import run
    from .policies import IrisPolicy, LLMActionPolicy

    if not config.WA_SHOPPING:
        print("ERROR: WA_SHOPPING not set (source scripts/export_webarena_env.sh)")
        return 1
    cls = IrisPolicy if args.policy == "iris" else LLMActionPolicy
    policy = cls(model=args.model or None, base_url=args.base_url or None,
                 api_key_env=args.api_key_env, max_tokens=args.max_tokens)
    renv = _make_live_renv(headless=not args.no_headless)
    try:
        run(policy, tag=args.tag, which=args.states, limit=args.limit,
            budget=args.budget,
            variants=tuple(v for v in args.variants.split(",") if v.strip()),
            renv=renv)
    finally:
        renv.close()
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
    s.set_defaults(fn=cmd_assemble)

    s = sub.add_parser("assemble-multiturn",
                       help="trajectories + key states -> multi-turn SFT/DPO")
    s.set_defaults(fn=cmd_assemble_multiturn)

    s = sub.add_parser("split", help="product-level train/test split")
    s.add_argument("--holdout-frac", type=float, default=0.25)
    s.set_defaults(fn=cmd_split)

    s = sub.add_parser("train-dpo", help="LoRA DPO on preference pairs (TRL)")
    s.add_argument("--data", default="", help="pairs jsonl (default splits/dpo_train)")
    s.add_argument("--adapter", default="",
                   help="SFT LoRA dir to warm-start from (merged; ref = SFT policy)")
    s.add_argument("--beta", type=float, default=0.1)
    s.add_argument("--max-steps", type=int, default=-1)
    s.add_argument("--output-dir", default="")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(fn=cmd_train_dpo)

    s = sub.add_parser("train-grpo",
                       help="GRPO/GSPO with grounded verifiable rewards (TRL)")
    s.add_argument("--data", default="",
                   help="comma-separated SFT jsonl paths (default: sft_train + multiturn)")
    s.add_argument("--gspo", action="store_true",
                   help="sequence-level importance sampling (GSPO)")
    s.add_argument("--adapter", default="",
                   help="SFT LoRA dir to warm-start the policy from (merged)")
    s.add_argument("--num-generations", type=int, default=4)
    s.add_argument("--max-steps", type=int, default=-1)
    s.add_argument("--output-dir", default="")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(fn=cmd_train_grpo)

    s = sub.add_parser("train", help="LoRA SFT (single-step and/or multi-turn)")
    s.add_argument("--data", default="",
                   help="sft jsonl (default splits/sft_train; multi-turn file works too)")
    s.add_argument("--max-len", type=int, default=0,
                   help="override train.max_len (multi-turn needs ~8192)")
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--output-dir", default="")
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("eval", help="decision accuracy on held-out split")
    s.add_argument("--adapter", default="")
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--max-new-tokens", type=int, default=200)
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("distill", help="teacher prose distillation (pinned labels)")
    s.add_argument("--limit", type=int, default=10)
    s.set_defaults(fn=cmd_distill)

    s = sub.add_parser("eval-rollout",
                       help="guarded live-rollout FSR eval (destructive risky "
                            "actions are recorded, never executed)")
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
    s.set_defaults(fn=cmd_eval_rollout)

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
