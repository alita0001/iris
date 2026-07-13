"""Admin-side probes (Magento admin / WebArena shopping_admin) — DRY-RUN
skeletons only in this phase.

Why they matter: the ADMIN privilege level offers exactly the reversibility
spectrum the customer side lacks — cancel order (compensable: record remains),
credit memo / refund (partial), disable product (reversible with cost),
delete product (irreversible data loss). Same site mechanics, different
privilege -> reversibility is a function of (action x mechanism x privilege).

All four probes here: navigate best-effort to the relevant admin screen,
locate the mutating affordance, record it as evidence, and STOP (UNKNOWN).
The commit paths are intentionally unimplemented until explicitly approved;
their `destructive` level is declared so `run_probe` will gate any future
commit request behind REVACT_ALLOW_DESTRUCTIVE as well.

Admin credentials: WebArena's sandbox admin login is public fixture data
(user `admin`), read from env WA_ADMIN_USER / WA_ADMIN_PASSWORD with the
WebArena defaults. These are NOT secrets in the API-key sense, but we still
never log the password.
"""
from __future__ import annotations

import os

from ...envs.obs_utils import find_bid_by_text
from .. import signals
from ..base import (DESTRUCTIVE, ProbeContext, ProbeSpec, ReversibilityResult,
                    mk_result, register)


def _admin_login(ctx: ProbeContext) -> bool:
    """Log into the Magento admin if we land on its login form."""
    if not ctx.admin_base:
        return False
    ctx.renv.step(f"goto('{ctx.admin_base}')")
    view = ctx.renv._last_obs_view
    user_el = find_bid_by_text(view, ["username"])
    if not user_el:                       # already logged in (dashboard)
        return True
    pass_el = find_bid_by_text(view, ["password"])
    sign_in = signals.find_action_by_text(view, ["sign in"], roles=("button",))
    if not (pass_el and sign_in):
        return False
    user = os.environ.get("WA_ADMIN_USER", "admin")
    password = os.environ.get("WA_ADMIN_PASSWORD", "admin1234")
    ctx.renv.step(f"fill('{user_el['bid']}', '{user}')")
    ctx.renv.step(f"fill('{pass_el['bid']}', '{password}')")
    ctx.renv.step(f"click('{sign_in['bid']}')")
    after = ctx.renv._last_obs_view
    return find_bid_by_text(after, ["username"]) is None


def _skeleton(ctx: ProbeContext, action_type: str, grounding: str,
              nav_path: str, affordance_subs: list[str]) -> ReversibilityResult:
    if not _admin_login(ctx):
        return mk_result(action_type, "UNKNOWN", grounding, destructive=False,
                         evidence={"reason": "admin login failed or admin_base unset"})
    ctx.renv.step(f"goto('{ctx.admin_base.rstrip('/')}{nav_path}')")
    view = ctx.renv._last_obs_view
    # open the first row if the grid links to a detail view
    row = signals.find_action_by_text(view, ["view"], roles=("link",))
    if row:
        ctx.renv.step(f"click('{row['bid']}')")
        view = ctx.renv._last_obs_view
    afford = signals.find_action_by_text(view, affordance_subs,
                                         roles=("button", "link"))
    return mk_result(
        action_type, "UNKNOWN", grounding, destructive=False,
        evidence={"reason": "dry-run skeleton (admin commit paths not implemented)",
                  "affordance_present": afford is not None,
                  "affordance": (afford or {}).get("line", ""),
                  "page_url": view.get("url", "")},
    )


def probe_admin_cancel_order(ctx: ProbeContext) -> ReversibilityResult:
    return _skeleton(ctx, "admin_cancel_order", "order_status",
                     "/sales/order/", ["cancel"])


def probe_admin_refund_order(ctx: ProbeContext) -> ReversibilityResult:
    return _skeleton(ctx, "admin_refund_order", "credit_memo_record",
                     "/sales/order/", ["credit memo", "refund"])


def probe_admin_disable_product(ctx: ProbeContext) -> ReversibilityResult:
    return _skeleton(ctx, "admin_disable_product", "product_enable_flag",
                     "/catalog/product/", ["enable product", "disable"])


def probe_admin_delete_product(ctx: ProbeContext) -> ReversibilityResult:
    return _skeleton(ctx, "admin_delete_product", "product_record",
                     "/catalog/product/", ["delete"])


register(ProbeSpec("shopping_admin.cancel_order", "shopping_admin",
                   "admin_cancel_order", DESTRUCTIVE, "order_status",
                   "reopen is impossible; cancel leaves a canceled record",
                   probe_admin_cancel_order,
                   expected_spectrum="PARTIALLY_RECOVERABLE"))
register(ProbeSpec("shopping_admin.refund_order", "shopping_admin",
                   "admin_refund_order", DESTRUCTIVE, "credit_memo_record",
                   "no undo; credit memo persists",
                   probe_admin_refund_order,
                   expected_spectrum="PARTIALLY_RECOVERABLE"))
register(ProbeSpec("shopping_admin.disable_product", "shopping_admin",
                   "admin_disable_product", DESTRUCTIVE, "product_enable_flag",
                   "re-enable the product",
                   probe_admin_disable_product,
                   expected_spectrum="REVERSIBLE"))
register(ProbeSpec("shopping_admin.delete_product", "shopping_admin",
                   "admin_delete_product", DESTRUCTIVE, "product_record",
                   "none (data loss)",
                   probe_admin_delete_product,
                   expected_spectrum="NOT_RECOVERED_WITHIN_BUDGET"))
