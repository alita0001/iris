"""Fingerprint: bid-invariance, restore detection, mutation detection."""
from revact.envs.fingerprint import StateFingerprint, fingerprint, is_restored


def _view(ax, url="http://x/a"):
    return {"url": url, "title": "t", "axtree_txt": ax}


def test_bids_are_stripped_from_hash():
    a = fingerprint(_view("[12] button 'Add to Cart'\n[15] link 'Home'"))
    b = fingerprint(_view("[99] button 'Add to Cart'\n[3] link 'Home'"))
    assert a.axtree_hash == b.axtree_hash
    assert is_restored(a, b)


def test_text_change_is_detected():
    a = fingerprint(_view("StaticText 'Cart (0)'"))
    b = fingerprint(_view("StaticText 'Cart (1)'"))
    assert not is_restored(a, b)


def test_backend_channel_is_authoritative():
    a = fingerprint({**_view("same"), "backend_state": {"cart": []}})
    b = fingerprint({**_view("same"), "backend_state": {"cart": ["x"]}})
    assert not is_restored(a, b)


def test_roundtrip_serialization():
    a = fingerprint(_view("[1] button 'X'"))
    b = StateFingerprint.from_dict(a.to_dict())
    assert is_restored(a, b)
