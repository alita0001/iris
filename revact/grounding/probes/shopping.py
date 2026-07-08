"""Customer-side shopping probes (Magento / WebArena shopping).

Non-destructive probes execute for real (their effects are transient by
construction); ``place_order`` and the dry-run skeletons are gated / never
click the mutating control.

Registered probes (see each spec's ``destructive`` level):
  shopping.add_to_cart            non_destructive   cart item count
  shopping.wishlist_add           non_destructive   wishlist item count
  shopping.compare_add            non_destructive   compare-list item count
  shopping.newsletter_subscribe   non_destructive   subscription checkbox
  shopping.currency_switch        non_destructive   currency switcher (N/A on
                                                    single-currency deployments)
  shopping.address_add            self_recovering   address count (adds then
                                                    deletes ITS OWN entry)
  shopping.place_order            destructive       order-number set (double-gated)
  shopping.review_submit          destructive       DRY-RUN skeleton only
  shopping.address_delete         destructive       DRY-RUN skeleton only
"""
from __future__ import annotations

from ... import config
from ...envs.obs_utils import find_bid_by_text
from .. import signals, undo
from ..base import (DESTRUCTIVE, NON_DESTRUCTIVE, SELF_RECOVERING, ProbeContext,
                    ProbeSpec, ReversibilityResult, mk_result, register)


# --------------------------------------------------------------------------- #
# Navigation helpers
# --------------------------------------------------------------------------- #
def add_item_to_cart(renv, product_url: str) -> None:
    """Add one unit, selecting any required option swatches first."""
    _o, _r, _t, _tr, _i, v = renv.step(f"goto('{product_url}')")
    for b in signals.radios(v):
        renv.step(f"click('{b}')")
    el = find_bid_by_text(renv._last_obs_view, ["add to cart"])
    if el:
        renv.step(f"click('{el['bid']}')")


def _click_product_link(renv, product_url: str, subs: list[str]) -> bool:
    """Open the product page and click a link matching `subs`; True on click."""
    renv.step(f"goto('{product_url}')")
    el = signals.find_action_by_text(renv._last_obs_view, subs)
    if not el:
        return False
    renv.step(f"click('{el['bid']}')")
    return True


def _drain_probe(action_type: str, grounding: str, sig_fn, drain_fn,
                 list_url: str, ctx: ProbeContext, do_action) -> ReversibilityResult:
    """Shared skeleton: clean baseline -> action -> signal -> undo -> compare."""
    base_sig = drain_fn(ctx.renv, list_url, ctx.budget)          # verified-clean baseline
    acted = do_action()
    if not acted:
        return mk_result(action_type, "UNKNOWN", grounding, destructive=False,
                         evidence={"reason": "action affordance not found"})
    after = sig_fn(ctx.renv, list_url)
    added = after["count"] > base_sig["count"]
    undone = drain_fn(ctx.renv, list_url, ctx.budget)            # undo controller
    restored = undone["count"] == base_sig["count"]
    label = ("NO_EFFECT" if not added else
             "REVERSIBLE" if restored else "PARTIALLY_RECOVERABLE")
    return mk_result(
        action_type, label, grounding, destructive=False,
        evidence={
            "baseline": base_sig["count"], "after_action": after["count"],
            "after_undo": undone["count"],
            "baseline_cleanup_steps": len(base_sig.get("undo_actions", [])),
            "undo_steps": len(undone.get("undo_actions", [])),
            "undo_actions": undone.get("undo_actions", []),
            "residual_diff": {"count_delta": undone["count"] - base_sig["count"]},
            "undo_budget_exhausted": undone.get("budget_exhausted", False),
        },
    )


# --------------------------------------------------------------------------- #
# Non-destructive probes
# --------------------------------------------------------------------------- #
def probe_add_to_cart(ctx: ProbeContext) -> ReversibilityResult:
    cart_url = ctx.base + config.SHOPPING_PATHS["cart"]
    return _drain_probe(
        "add_to_cart", "cart_item_count", signals.cart_signal, undo.empty_cart,
        cart_url, ctx,
        do_action=lambda: (add_item_to_cart(ctx.renv, ctx.product_url) or True),
    )


def probe_wishlist_add(ctx: ProbeContext) -> ReversibilityResult:
    wl_url = ctx.base + config.SHOPPING_PATHS["wishlist"]
    return _drain_probe(
        "wishlist_add", "wishlist_item_count", signals.wishlist_signal,
        undo.empty_wishlist, wl_url, ctx,
        # exact phrase only: "wish list" alone would match the 'My Wish List'
        # nav link, which appears earlier in document order than the add control
        do_action=lambda: _click_product_link(ctx.renv, ctx.product_url,
                                              ["add to wish list"]),
    )


def probe_compare_add(ctx: ProbeContext) -> ReversibilityResult:
    cmp_url = ctx.base + config.SHOPPING_PATHS["compare"]
    return _drain_probe(
        "compare_add", "compare_item_count", signals.compare_signal,
        undo.empty_compare, cmp_url, ctx,
        do_action=lambda: _click_product_link(ctx.renv, ctx.product_url,
                                              ["add to compare"]),
    )


def probe_newsletter_subscribe(ctx: ProbeContext) -> ReversibilityResult:
    manage = ctx.base + config.SHOPPING_PATHS["newsletter"]
    baseline = signals.newsletter_signal(ctx.renv, manage)
    if not baseline["found"]:
        return mk_result("newsletter_subscribe", "UNKNOWN", "subscription_checkbox",
                         destructive=False,
                         evidence={"reason": "subscription checkbox not found"})
    target = not baseline["checked"]
    flipped = undo.set_newsletter(ctx.renv, manage, subscribed=target)  # the action
    changed = flipped["checked"] == target and flipped["checked"] != baseline["checked"]
    restored_sig = undo.set_newsletter(ctx.renv, manage, subscribed=baseline["checked"])
    restored = restored_sig["checked"] == baseline["checked"]
    label = ("NO_EFFECT" if not changed else
             "REVERSIBLE" if restored else "PARTIALLY_RECOVERABLE")
    return mk_result(
        "newsletter_subscribe", label, "subscription_checkbox", destructive=False,
        evidence={
            "baseline_checked": baseline["checked"],
            "after_action_checked": flipped["checked"],
            "after_undo_checked": restored_sig["checked"],
            "undo_steps": len(restored_sig.get("undo_actions", [])),
            "undo_actions": restored_sig.get("undo_actions", []),
            "residual_diff": {"checked_changed": restored_sig["checked"] != baseline["checked"]},
        },
    )


def probe_currency_switch(ctx: ProbeContext) -> ReversibilityResult:
    """Single-currency deployments have no switcher -> honest UNKNOWN(N/A)."""
    ctx.renv.step(f"goto('{ctx.base}')")
    el = signals.find_action_by_text(ctx.renv._last_obs_view,
                                     ["currency", "usd", "eur"], roles=("button", "link", "combobox"))
    if not el:
        return mk_result("currency_switch", "UNKNOWN", "currency_selector",
                         destructive=False,
                         evidence={"reason": "no currency switcher on this deployment "
                                             "(single-currency store)"})
    return mk_result("currency_switch", "UNKNOWN", "currency_selector",
                     destructive=False,
                     evidence={"reason": "switcher found but probe not implemented",
                               "switcher": el["line"]})


# --------------------------------------------------------------------------- #
# Self-recovering probe: add an address, then delete the one WE added
# --------------------------------------------------------------------------- #
_ADDRESS_FIELDS = [
    (["first name"], "Rev"),
    (["last name"], "Act"),
    (["company"], ""),                       # optional; skipped when value == ""
    (["phone number", "telephone"], "5555550100"),
    (["street address: line 1", "street address"], "1 Probe Street"),
    (["city"], "Springfield"),
    (["zip", "postal"], "60601"),
]


def probe_address_add(ctx: ProbeContext) -> ReversibilityResult:
    addr_url = ctx.base + config.SHOPPING_PATHS["address"]
    new_url = ctx.base + config.SHOPPING_PATHS["address_new"]
    baseline = signals.address_signal(ctx.renv, addr_url)

    _o, _r, _t, _tr, _i, v = ctx.renv.step(f"goto('{new_url}')")
    filled = []
    for labels, value in _ADDRESS_FIELDS:
        if not value:
            continue
        el = find_bid_by_text(ctx.renv._last_obs_view, labels)
        if el:
            ctx.renv.step(f"fill('{el['bid']}', '{value}')")
            filled.append(labels[0])
    # State/Province combobox (required for US)
    state_el = signals.find_action_by_text(ctx.renv._last_obs_view,
                                           ["state/province", "state"],
                                           roles=("combobox",))
    if state_el:
        ctx.renv.step(f"select_option('{state_el['bid']}', 'Illinois')")
    save = signals.find_action_by_text(ctx.renv._last_obs_view, ["save address"])
    if not save:
        return mk_result("address_add", "UNKNOWN", "address_count", destructive=False,
                         evidence={"reason": "Save Address button not found",
                                   "filled": filled})
    ctx.renv.step(f"click('{save['bid']}')")

    after = signals.address_signal(ctx.renv, addr_url)
    added = after["count"] > baseline["count"]
    if not added:
        return mk_result("address_add", "UNKNOWN", "address_count", destructive=False,
                         evidence={"reason": "save did not create an entry "
                                             "(form validation?)", "filled": filled})
    undone = undo.delete_last_address(ctx.renv, addr_url)     # deletes OUR entry only
    restored = undone["count"] == baseline["count"]
    label = "REVERSIBLE" if restored else "PARTIALLY_RECOVERABLE"
    return mk_result(
        "address_add", label, "address_count", destructive=False,
        evidence={
            "baseline": baseline["count"], "after_action": after["count"],
            "after_undo": undone["count"], "filled": filled,
            "undo_steps": len(undone.get("undo_actions", [])),
            "undo_actions": undone.get("undo_actions", []),
            "residual_diff": {"count_delta": undone["count"] - baseline["count"]},
        },
    )


# --------------------------------------------------------------------------- #
# Destructive probes (double-gated / dry-run skeletons)
# --------------------------------------------------------------------------- #
def probe_place_order(ctx: ProbeContext) -> ReversibilityResult:
    """Places ONE real order when commit is granted (run_probe enforces the
    REVACT_ALLOW_DESTRUCTIVE gate before this function ever sees commit=True)."""
    hist = ctx.base + config.SHOPPING_PATHS["order_history"]
    checkout = ctx.base + config.SHOPPING_PATHS["checkout"]
    before = signals.order_ids(ctx.renv, hist)

    add_item_to_cart(ctx.renv, ctx.product_url)
    _o, _r, _t, _tr, _i, v = ctx.renv.step(f"goto('{checkout}')")
    for b in signals.radios(v):                                # shipping method
        ctx.renv.step(f"click('{b}')")
    nxt = find_bid_by_text(ctx.renv._last_obs_view, ["next"])
    if nxt:
        ctx.renv.step(f"click('{nxt['bid']}')")                # -> payment step
    po = find_bid_by_text(ctx.renv._last_obs_view, ["place order"])
    if po is None:
        return mk_result("place_order", "UNKNOWN", "order_set", destructive=False,
                         evidence={"reason": "no Place Order reached"})
    if not ctx.commit:
        undo.empty_cart(ctx.renv, ctx.base + config.SHOPPING_PATHS["cart"])
        return mk_result("place_order", "UNKNOWN", "order_set", destructive=False,
                         evidence={"reason": "dry-run (no commit); Place Order "
                                             "reached, cart cleaned up"})

    ctx.renv.step(f"click('{po['bid']}')")                     # DESTRUCTIVE
    after = signals.order_ids(ctx.renv, hist)
    new = sorted(after - before)
    cancel_el = None
    if new:
        _o, _r, _t, _tr, _i, v = ctx.renv.step(f"goto('{hist}')")
        cancel_el = signals.find_action_by_text(v, ["cancel order", "cancel this", "cancel"])
    cancelable = cancel_el is not None
    label = ("NO_EFFECT" if not new else
             "PARTIALLY_RECOVERABLE" if cancelable else "IRREVERSIBLE")
    return mk_result(
        "place_order", label, "order_set", destructive=True,
        evidence={
            "new_orders": new, "customer_cancelable": cancelable,
            "cancel_match": (cancel_el or {}).get("line", ""),
            "undo_steps": 0, "undo_actions": [],
            "residual_diff": {"orders_added": new},
        },
        commit_mode=True,
    )


def probe_review_submit(ctx: ProbeContext) -> ReversibilityResult:
    """DRY-RUN skeleton: reach the review form, locate Submit Review, stop.
    The commit path is intentionally NOT implemented (a posted review has no
    customer-side delete; grounding it would leave a lasting record)."""
    ctx.renv.step(f"goto('{ctx.product_url}')")
    link = signals.find_action_by_text(ctx.renv._last_obs_view,
                                       ["add your review", "be the first to review",
                                        "write a review", "reviews"])
    if link:
        ctx.renv.step(f"click('{link['bid']}')")
    submit = signals.find_action_by_text(ctx.renv._last_obs_view, ["submit review"])
    return mk_result(
        "review_submit", "UNKNOWN", "review_record", destructive=False,
        evidence={"reason": "dry-run skeleton (commit path not implemented)",
                  "form_reached": submit is not None,
                  "submit_control": (submit or {}).get("line", "")},
    )


def probe_address_delete(ctx: ProbeContext) -> ReversibilityResult:
    """DRY-RUN skeleton: record whether a delete affordance exists on the
    address book. Deleting a PRE-EXISTING address is destructive (content is
    lost even though an address can be re-added), so no commit path here."""
    addr_url = ctx.base + config.SHOPPING_PATHS["address"]
    sig = signals.address_signal(ctx.renv, addr_url)
    return mk_result(
        "address_delete", "UNKNOWN", "address_count", destructive=False,
        evidence={"reason": "dry-run skeleton (deleting pre-existing data)",
                  "deletable_entries": sig["count"],
                  "affordance_present": sig["count"] > 0},
    )


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
register(ProbeSpec("shopping.add_to_cart", "shopping", "add_to_cart",
                   NON_DESTRUCTIVE, "cart_item_count", "remove each cart line",
                   probe_add_to_cart, expected_spectrum="REVERSIBLE"))
register(ProbeSpec("shopping.wishlist_add", "shopping", "wishlist_add",
                   NON_DESTRUCTIVE, "wishlist_item_count", "remove each wishlist item",
                   probe_wishlist_add, expected_spectrum="REVERSIBLE"))
register(ProbeSpec("shopping.compare_add", "shopping", "compare_add",
                   NON_DESTRUCTIVE, "compare_item_count", "remove each compare item",
                   probe_compare_add, expected_spectrum="REVERSIBLE"))
register(ProbeSpec("shopping.newsletter_subscribe", "shopping", "newsletter_subscribe",
                   NON_DESTRUCTIVE, "subscription_checkbox", "toggle back and save",
                   probe_newsletter_subscribe, expected_spectrum="REVERSIBLE"))
register(ProbeSpec("shopping.currency_switch", "shopping", "currency_switch",
                   NON_DESTRUCTIVE, "currency_selector", "switch back",
                   probe_currency_switch, expected_spectrum="REVERSIBLE"))
register(ProbeSpec("shopping.address_add", "shopping", "address_add",
                   SELF_RECOVERING, "address_count", "delete the entry the probe added",
                   probe_address_add, expected_spectrum="REVERSIBLE"))
register(ProbeSpec("shopping.place_order", "shopping", "place_order",
                   DESTRUCTIVE, "order_set", "look for an actionable customer cancel",
                   probe_place_order, expected_spectrum="IRREVERSIBLE"))
register(ProbeSpec("shopping.review_submit", "shopping", "review_submit",
                   DESTRUCTIVE, "review_record", "none available customer-side",
                   probe_review_submit, expected_spectrum="IRREVERSIBLE"))
register(ProbeSpec("shopping.address_delete", "shopping", "address_delete",
                   DESTRUCTIVE, "address_count", "re-add is possible but content is lost",
                   probe_address_delete, expected_spectrum="PARTIALLY_RECOVERABLE"))
