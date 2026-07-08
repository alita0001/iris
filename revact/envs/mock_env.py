"""Offline MockShoppingEnv: a deterministic shopping state machine.

Purpose: exercise and unit-test the harness AND the grounding probes (signals,
undo controllers, probe protocol) WITHOUT WebArena/Docker/Playwright, using
plain stdlib. It mimics the gymnasium interface used by BrowserGym:

    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step("click('30')")
    env.close()

Page markup intentionally mirrors Magento's a11y phrasing (``link 'Remove
item'``, "You have no items in your shopping cart", ``gridcell '000001001'``
order numbers, wishlist / newsletter pages) so that
``revact.grounding.signals`` works on the mock exactly as on the live site.
NOT a WebArena substitute for final experiments — only a pipeline test-bed.
"""
from __future__ import annotations

import copy
import re

_CLICK_RE = re.compile(r"click\(\s*['\"](\d+)['\"]\s*\)")
_GOTO_RE = re.compile(r"goto\(\s*['\"]([^'\"]+)['\"]\s*\)")
_BASE = "http://mock-shopping"

# Initial backend: one address on file, empty cart/wishlist, no orders,
# newsletter unsubscribed.
_INIT_BACKEND = {
    "cart": [],
    "orders": [],
    "wishlist": [],
    "newsletter": False,
    "addresses": [{"id": 1, "line": "123 Main St, Springfield"}],
}


def _page_home() -> str:
    return (
        "RootWebArea 'Mock Shop Home'\n"
        "[10] textbox 'Search'\n"
        "[11] link 'Shop Laptops'\n"
        "[12] link 'My Account Addresses'"
    )


def _page_list() -> str:
    return (
        "RootWebArea 'Laptops'\n"
        "[11] link 'Shop Laptops'\n"
        "[20] link 'Dell XPS 13'\n"
        "[21] link 'Lenovo ThinkPad'"
    )


def _page_product(cart_n: int) -> str:
    return (
        "RootWebArea 'Dell XPS 13'\n"
        "[30] button 'Add to Cart'\n"
        "[32] link 'Add to Wish List'\n"
        "[31] link 'Back to results'\n"
        f"StaticText 'Cart ({cart_n})'"
    )


def _page_cart(cart) -> str:
    lines = ["RootWebArea 'Shopping Cart'"]
    if not cart:
        lines.append("StaticText 'You have no items in your shopping cart.'")
    for i, item in enumerate(cart):
        lines.append(f"StaticText '{item}'")
        lines.append(f"[{45 + i}] link 'Remove item'")
    lines.append("[40] button 'Proceed to Checkout'")
    lines.append(f"StaticText 'Cart ({len(cart)})'")
    return "\n".join(lines)


def _page_checkout(cart) -> str:
    items = ", ".join(cart) if cart else "(empty)"
    return (
        "RootWebArea 'Checkout'\n"
        f"StaticText 'Items: {items}'\n"
        "[50] button 'Place Order'\n"
        "[41] link 'Back to Cart'"
    )


def _page_order_confirmation(order_id: int) -> str:
    return (
        "RootWebArea 'Order Confirmation'\n"
        f"StaticText 'Order # {order_id} placed successfully'\n"
        "[70] link 'Continue Shopping'"
    )


def _page_order_history(orders) -> str:
    lines = ["RootWebArea 'My Orders'"]
    if not orders:
        lines.append("StaticText 'You have placed no orders.'")
    for i, oid in enumerate(orders):
        # Magento renders order numbers as 9-digit gridcells.
        lines.append(f"[{75 + i}] gridcell '{oid:09d}'")
        lines.append(f"[{85 + i}] link 'View Order'")
    return "\n".join(lines)


def _page_wishlist(wishlist) -> str:
    lines = ["RootWebArea 'My Wish List'"]
    if not wishlist:
        lines.append("StaticText 'You have no items in your wish list.'")
    for i, item in enumerate(wishlist):
        lines.append(f"StaticText '{item}'")
        lines.append(f"[{55 + i}] link 'Remove item'")
    return "\n".join(lines)


def _page_newsletter(pending: bool) -> str:
    checked = ", checked" if pending else ""
    return (
        "RootWebArea 'Newsletter Subscription'\n"
        f"[90] checkbox 'General Subscription'{checked}\n"
        "[91] button 'Save'"
    )


def _page_addresses(addresses) -> str:
    lines = ["RootWebArea 'Address Book'"]
    for a in addresses:
        lines.append(f"StaticText '{a['line']}'")
        lines.append(f"[{60 + a['id']}] button 'Delete'")
    lines.append("[65] button 'Add New Address'")
    return "\n".join(lines)


class MockShoppingEnv:
    """Deterministic; identical (seed, action-sequence) -> identical state."""

    def __init__(self, goal: str = "Explore the shop.", max_steps: int = 60,
                 reward_fn=None):
        self.goal = goal
        self.max_steps = max_steps
        # reward_fn(backend_state, page, goal) -> float; default: always 0.
        self.reward_fn = reward_fn
        self._backend = None
        self._page = None
        self._url = None
        self._nav_stack: list[tuple[str, str]] = []
        self._steps = 0
        self._next_order_id = 1001
        self._newsletter_pending = False

    # -- gymnasium-like API ------------------------------------------------ #
    def reset(self, seed: int = 0):
        self._backend = copy.deepcopy(_INIT_BACKEND)
        self._page = "home"
        self._url = f"{_BASE}/"
        self._nav_stack = []
        self._steps = 0
        self._next_order_id = 1001
        self._newsletter_pending = self._backend["newsletter"]
        return self._obs(), {}

    def step(self, action: str):
        self._steps += 1
        self._apply(action)
        reward = 0.0
        if self.reward_fn is not None:
            reward = float(self.reward_fn(self._backend, self._page, self.goal))
        terminated = reward >= 1.0
        truncated = self._steps >= self.max_steps
        return self._obs(), reward, terminated, truncated, {}

    def close(self):  # noqa: D401 - parity with gym
        return None

    # -- internals --------------------------------------------------------- #
    def _goto(self, page: str, url: str):
        self._nav_stack.append((self._page, self._url))
        self._page, self._url = page, url

    def _apply(self, action: str):
        action = (action or "").strip()
        if action.startswith("go_back"):
            if self._nav_stack:
                self._page, self._url = self._nav_stack.pop()
            return
        mg = _GOTO_RE.search(action)
        if mg:
            url = mg.group(1)
            if "customer/address" in url:
                self._goto("addresses", url)
            elif "checkout/cart" in url:
                self._goto("cart", url)
            elif "sales/order/history" in url:
                self._goto("order_history", url)
            elif "wishlist" in url:
                self._goto("wishlist", url)
            elif "newsletter" in url:
                self._goto("newsletter", url)
            elif url.rstrip("/").endswith("/checkout"):
                # checkout requires a non-empty cart, else it stays on cart
                self._goto("checkout" if self._backend["cart"] else "cart", url)
            elif "/product/" in url or url.endswith(".html"):
                self._goto("product", url)
            else:
                self._goto("home", url)
            return
        m = _CLICK_RE.search(action)
        if not m:
            return  # no-op (e.g. focusing a textbox)
        bid = m.group(1)
        b = self._backend

        if self._page == "home":
            if bid == "11":
                self._goto("list", f"{_BASE}/laptops")
            elif bid == "12":
                self._goto("addresses", f"{_BASE}/account/addresses")
        elif self._page == "list":
            if bid in ("20", "21"):
                self._goto("product", f"{_BASE}/product/{bid}")
        elif self._page == "product":
            if bid == "30":
                b["cart"].append("Dell XPS 13")
                self._goto("cart", f"{_BASE}/cart")
            elif bid == "32":
                b["wishlist"].append("Dell XPS 13")
                self._goto("wishlist", f"{_BASE}/wishlist/")
            elif bid == "31":
                self._apply("go_back()")
        elif self._page == "cart":
            if bid == "40":
                self._goto("checkout", f"{_BASE}/checkout")
            elif bid.isdigit() and int(bid) >= 45:
                idx = int(bid) - 45
                if 0 <= idx < len(b["cart"]):
                    b["cart"].pop(idx)
        elif self._page == "wishlist":
            if bid.isdigit() and int(bid) >= 55:
                idx = int(bid) - 55
                if 0 <= idx < len(b["wishlist"]):
                    b["wishlist"].pop(idx)
        elif self._page == "newsletter":
            if bid == "90":
                self._newsletter_pending = not self._newsletter_pending
            elif bid == "91":
                b["newsletter"] = self._newsletter_pending
        elif self._page == "checkout":
            if bid == "50":
                oid = self._next_order_id
                self._next_order_id += 1
                b["orders"].append(oid)
                b["cart"] = []
                self._goto("order_confirmation", f"{_BASE}/order/{oid}")
            elif bid == "41":
                self._apply("go_back()")
        elif self._page == "addresses":
            if bid.isdigit() and int(bid) >= 61:
                aid = int(bid) - 60
                b["addresses"] = [a for a in b["addresses"] if a["id"] != aid]
            elif bid == "65":
                new_id = (max((a["id"] for a in b["addresses"]), default=0)) + 1
                b["addresses"].append({"id": new_id, "line": f"New Address #{new_id}"})

    def _render(self) -> str:
        b = self._backend
        if self._page == "home":
            return _page_home()
        if self._page == "list":
            return _page_list()
        if self._page == "product":
            return _page_product(len(b["cart"]))
        if self._page == "cart":
            return _page_cart(b["cart"])
        if self._page == "checkout":
            return _page_checkout(b["cart"])
        if self._page == "order_confirmation":
            return _page_order_confirmation(b["orders"][-1] if b["orders"] else 0)
        if self._page == "order_history":
            return _page_order_history(b["orders"])
        if self._page == "wishlist":
            return _page_wishlist(b["wishlist"])
        if self._page == "newsletter":
            return _page_newsletter(self._newsletter_pending)
        if self._page == "addresses":
            return _page_addresses(b["addresses"])
        return "RootWebArea 'Unknown'"

    def _obs(self) -> dict:
        return {
            "goal": self.goal,
            "url": self._url,
            "open_pages_urls": [self._url],
            "active_page_index": 0,
            "axtree_txt": self._render(),
            "backend_state": copy.deepcopy(self._backend),
        }


# =========================================================================== #
# MockRedditEnv: a deterministic Postmill (WebArena reddit) state machine.
#
# Mirrors Postmill's a11y phrasing so revact.grounding.signals works on the mock
# exactly as on the live site: a submission vote widget ('Upvote'/'Downvote' +
# 'Score: N'), a per-forum 'Subscribe'/'Unsubscribe' toggle, a comment box +
# 'Post' button, per-(own-)comment 'Delete' links, and a '[deleted]' tombstone
# left behind when a posted comment is removed (the cross-site analogue of the
# shopping 'Canceled' status-text confound). NOT a WebArena substitute — a
# pipeline/probe test-bed only.
# =========================================================================== #
_R_BASE = "http://mock-reddit"
_FILL_RE = re.compile(r"fill\(\s*['\"](\d+)['\"]\s*,\s*['\"](.*)['\"]\s*\)")

_R_INIT_BACKEND = {
    # forum -> subscribed?
    "subscriptions": {"books": True, "AskReddit": False},
    # submission id -> {"forum","title","base_score","my_vote"(-1/0/1),
    #                   "comments":[{"marker","deleted"}]}
    "submissions": {
        "1": {"forum": "AskReddit", "title": "What is your favorite book?",
              "base_score": 38, "my_vote": 0, "comments": []},
    },
}


def _r_page_forum(name: str, subscribed: bool) -> str:
    toggle = "Unsubscribe" if subscribed else "Subscribe"
    return (
        f"RootWebArea 'f/{name}'\n"
        f"[210] button '{toggle}'\n"
        "[211] link 'What is your favorite book? (3 comments)'\n"
        "[212] link 'Submit'"
    )


def _r_page_submission(sub: dict) -> str:
    score = sub["base_score"] + sub["my_vote"]
    # Postmill relabels the up-control to 'Retract upvote' once the user upvotes
    # (both end in "upvote"); signals._submission_vote reads this active state.
    up_label = "Retract upvote" if sub["my_vote"] == 1 else "Upvote"
    lines = [
        f"RootWebArea '{sub['title']}'",
        # Postmill order: up-control, the net score, Downvote (score sits BETWEEN
        # the two vote buttons — signals.vote_score relies on this order).
        f"[220] button '{up_label}'",
        f"StaticText 'Score: {score}'",
        "[221] button 'Downvote'",
        "[222] textbox 'Comment'",
        "[223] button 'Post'",
    ]
    bid = 230
    for c in sub["comments"]:
        if c.get("deleted"):
            lines.append("StaticText '[deleted]'")
        else:
            lines.append(f"StaticText '{c['marker']}'")
            lines.append(f"[{bid}] link 'Delete'")
            bid += 1
    return "\n".join(lines)


class MockRedditEnv:
    """Deterministic Postmill mock; identical (seed, actions) -> identical state."""

    def __init__(self, goal: str = "Browse the forums.", max_steps: int = 60,
                 reward_fn=None):
        self.goal = goal
        self.max_steps = max_steps
        self.reward_fn = reward_fn
        self._backend = None
        self._page = None          # ("forum", name) | ("submission", id) | ("home",)
        self._url = None
        self._steps = 0
        self._pending_comment = ""     # value filled into the comment box

    def reset(self, seed: int = 0):
        self._backend = copy.deepcopy(_R_INIT_BACKEND)
        self._page = ("home",)
        self._url = f"{_R_BASE}/"
        self._steps = 0
        self._pending_comment = ""
        return self._obs(), {}

    def step(self, action: str):
        self._steps += 1
        self._apply(action)
        reward = 0.0
        if self.reward_fn is not None:
            reward = float(self.reward_fn(self._backend, self._page, self.goal))
        return self._obs(), reward, reward >= 1.0, self._steps >= self.max_steps, {}

    def close(self):
        return None

    # -- internals --------------------------------------------------------- #
    def _apply(self, action: str):
        action = (action or "").strip()
        mg = _GOTO_RE.search(action)
        if mg:
            url = mg.group(1)
            m_sub = re.search(r"/f/[^/]+/(\d+)", url)
            m_forum = re.search(r"/f/([^/]+)/?$", url)
            if m_sub and m_sub.group(1) in self._backend["submissions"]:
                self._page, self._url = ("submission", m_sub.group(1)), url
            elif m_forum:
                self._page, self._url = ("forum", m_forum.group(1)), url
            else:
                self._page, self._url = ("home",), url
            return
        mf = _FILL_RE.search(action)
        if mf:
            if mf.group(1) == "222":
                self._pending_comment = mf.group(2)
            return
        m = _CLICK_RE.search(action)
        if not m:
            return
        bid = m.group(1)
        b = self._backend
        if self._page[0] == "forum":
            name = self._page[1]
            if bid == "210":                       # subscribe/unsubscribe toggle
                b["subscriptions"][name] = not b["subscriptions"].get(name, False)
            elif bid == "211":
                self._page, self._url = ("submission", "1"), f"{_R_BASE}/f/{name}/1"
        elif self._page[0] == "submission":
            sub = b["submissions"][self._page[1]]
            if bid == "220":                       # upvote toggles
                sub["my_vote"] = 0 if sub["my_vote"] == 1 else 1
            elif bid == "221":                     # downvote toggles
                sub["my_vote"] = 0 if sub["my_vote"] == -1 else -1
            elif bid == "223":                     # post comment
                if self._pending_comment:
                    sub["comments"].append({"marker": self._pending_comment,
                                            "deleted": False})
                    self._pending_comment = ""
            elif bid.isdigit() and int(bid) >= 230:
                # Delete the Nth live (non-deleted) comment -> leaves a tombstone.
                live = [c for c in sub["comments"] if not c.get("deleted")]
                idx = int(bid) - 230
                if 0 <= idx < len(live):
                    live[idx]["deleted"] = True

    def _render(self) -> str:
        b = self._backend
        if self._page[0] == "forum":
            name = self._page[1]
            return _r_page_forum(name, b["subscriptions"].get(name, False))
        if self._page[0] == "submission":
            return _r_page_submission(b["submissions"][self._page[1]])
        return ("RootWebArea 'Postmill'\n"
                "[200] link 'Forums'\n"
                "[201] link 'f/books'\n"
                "[202] link 'f/AskReddit'")

    def _obs(self) -> dict:
        return {
            "goal": self.goal,
            "url": self._url,
            "open_pages_urls": [self._url],
            "active_page_index": 0,
            "axtree_txt": self._render(),
            "backend_state": copy.deepcopy(self._backend),
        }
