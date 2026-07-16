from revact.envs import webarena_patch


def test_openrouter_is_a_configurable_judge_route_alias(monkeypatch):
    calls = []
    monkeypatch.setenv("REVACT_WA_JUDGE", "openrouter")
    monkeypatch.setattr(
        webarena_patch, "route_llm_reward",
        lambda **_kwargs: calls.append("route") or True)
    assert webarena_patch.configure_reward_judge(verbose=False) == "route"
    assert calls == ["route"]
