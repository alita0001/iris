import json
from dataclasses import replace

import pytest

from revact.data import opinions as opinions_module
from revact.data.opinions import (
    NOT_RISKY,
    OPINION_ARTIFACT_ROLE,
    OPINION_IMPORT_SCHEMA_VERSION,
    OPINION_LABEL_SCHEMA_VERSION,
    PERCEIVED_CHANGE,
    PERCEIVED_EFFECTS,
    PERCEIVED_NO_CHANGE,
    PERCEIVED_NOT_RECOVERABLE,
    PERCEIVED_RECOVERABLE,
    RATER_HUMAN,
    RATER_LLM,
    RISKY,
    OpinionLabelError,
    OpinionLabelRecord,
    assert_opinion_manifest_integrity,
    build_behavior_grounding_disagreement_matrix,
    build_point_rater_matrix,
    import_external_opinion_ratings,
    import_opinion_records,
    load_opinion_records,
    make_opinion_label_id,
    save_opinion_records,
)
from revact.eval.truth import EvaluationTruthError, EvaluationTruthRecord
from revact.grounding.schema import (
    EFFECT_CHANGED,
    RECOVERY_RECOVERED,
    GroundingPoint,
    GroundingValidationError,
)


def _point(point_id="point-1", state_id="state-1") -> GroundingPoint:
    return GroundingPoint(
        probe_point_id=point_id,
        probe_run_id="probe-run-1",
        probe_name="shopping.add_to_cart",
        state_id=state_id,
        candidate_id=f"candidate-{point_id}",
        action_instance_id=f"action-{point_id}",
        action_type="add_to_cart",
        raw_action="click('42')",
        canonical_action="click:add_to_cart:sku-1",
        site="shopping",
        environment_family="webarena",
        environment_instance="shopping:7770",
        environment_origin="webarena",
        is_mock=False,
        task_id="webarena.1",
        trajectory_id=f"traj-{point_id}",
        run_id="run-1",
        seed=0,
        url="http://shopping/product/1",
        account="user-1",
        privilege="customer",
        budget_k=12,
        solver_set=["site_specific_deterministic"],
        controller_version="test-controller",
        pre_observation_hash="pre-hash",
        pre_signal={"cart_count": 0},
        post_observation_hash="post-hash",
        post_signal={"cart_count": 1},
        undo_actions=["click('remove-1')"],
        undo_semantic_actions=["remove_cart_item(sku-1)"],
        undo_observation_hashes=["undo-hash"],
        final_signal={"cart_count": 0},
        effect_status=EFFECT_CHANGED,
        recovery_status=RECOVERY_RECOVERED,
        undo_cost_steps=1,
        residual_diff={},
        budget_exhausted=False,
        timestamp="2026-07-14T00:00:00+00:00",
        code_version="deadbeef",
        evidence={
            "measurement": "fixture",
            "candidate_snapshot_hash": "candidate-snapshot-hash",
        },
    )


def _opinion(
    *,
    point_id="point-1",
    state_id="state-1",
    rater_id="human-001",
    rater_type=RATER_HUMAN,
    provider="direct-study",
    model=None,
    effect=PERCEIVED_CHANGE,
    recovery=PERCEIVED_RECOVERABLE,
    risk=NOT_RISKY,
    evaluation_case_id=None,
    variant="constraint",
):
    instrument_id = "iris-opinion-instrument"
    instrument_version = "v1"
    evaluation_case_id = evaluation_case_id or point_id.replace("point", "case")
    goal_sha256 = (point_id[-1] * 64) if point_id[-1].isdigit() else "a" * 64
    input_sha256 = ("b" if point_id == "point-1" else "c") * 64
    raw_response = json.dumps({
        "effect": effect, "recovery": recovery, "normative_risk": risk,
        "confidence": 0.8,
        "rationale": "Independent rating from the displayed state and action.",
    })
    return OpinionLabelRecord(
        schema_version=OPINION_LABEL_SCHEMA_VERSION,
        artifact_role=OPINION_ARTIFACT_ROLE,
        opinion_label_id=make_opinion_label_id(
            point_id, evaluation_case_id, goal_sha256, rater_id,
            instrument_id, instrument_version
        ),
        probe_point_id=point_id,
        state_id=state_id,
        evaluation_case_id=evaluation_case_id,
        variant=variant,
        goal_sha256=goal_sha256,
        opinion_input_sha256=input_sha256,
        input_messages_sha256="d" * 64,
        raw_response=raw_response,
        raw_response_sha256=__import__("hashlib").sha256(
            raw_response.encode()).hexdigest(),
        provider_response_id="response-fixture",
        response_model=model or "human",
        finish_reason="stop" if model else "human_submission",
        rater_id=rater_id,
        rater_type=rater_type,
        provider=provider,
        model=model,
        prompt_generation_fp="instrument-prompt-fp-v1",
        instrument_id=instrument_id,
        instrument_version=instrument_version,
        perceived_effect=effect,
        perceived_recoverability=recovery,
        normative_risk_opinion=risk,
        confidence=0.8,
        rationale="Independent rating from the displayed state and action.",
        source_record_id=f"external-{point_id}-{rater_id}",
        collection_timestamp="2026-07-14T00:00:00+00:00",
        import_batch_id="opinion-import-001",
        code_version="deadbeef",
    )


def test_point_keyed_round_trip_manifest_and_immutable_append(tmp_path):
    point = _point()
    body = tmp_path / "opinion_labels.v1.jsonl"
    manifest = tmp_path / "OPINION_LABEL_MANIFEST.v1.jsonl"
    record = _opinion()

    save_opinion_records(
        [record], body, manifest, points={point.probe_point_id: point}
    )
    assert body.read_text().count("\n") == manifest.read_text().count("\n") == 1
    assert load_opinion_records(body)[record.opinion_label_id] == record
    assert_opinion_manifest_integrity(
        body, manifest, points={point.probe_point_id: point}
    )

    # Byte-identical rematerialization is idempotent; a changed body cannot be
    # silently substituted without append/versioning.
    save_opinion_records(
        [record], body, manifest, points={point.probe_point_id: point}
    )
    with pytest.raises(OpinionLabelError, match="refusing to overwrite"):
        save_opinion_records(
            [_opinion(risk=RISKY)],
            body,
            manifest,
            points={point.probe_point_id: point},
        )

    save_opinion_records(
        [record], body, manifest, append=True, points={point.probe_point_id: point}
    )
    assert body.read_text().count("\n") == 1
    with pytest.raises(OpinionLabelError, match="immutable opinion collision"):
        save_opinion_records(
            [_opinion(risk=RISKY)],
            body,
            manifest,
            append=True,
            points={point.probe_point_id: point},
        )


def test_manifest_tamper_or_torn_pair_fails_closed(tmp_path):
    body = tmp_path / "opinions.jsonl"
    manifest = tmp_path / "opinions.manifest.jsonl"
    save_opinion_records([_opinion()], body, manifest)
    row = json.loads(manifest.read_text())
    row["record_sha256"] = "0" * 64
    manifest.write_text(json.dumps(row) + "\n")
    with pytest.raises(OpinionLabelError, match="hash mismatch"):
        assert_opinion_manifest_integrity(body, manifest)

    manifest.unlink()
    with pytest.raises(OpinionLabelError, match="both exist or both be absent"):
        save_opinion_records([_opinion()], body, manifest, append=True)


def test_pair_write_rolls_back_when_second_rename_fails(tmp_path, monkeypatch):
    body = tmp_path / "opinions.jsonl"
    manifest = tmp_path / "opinions.manifest.jsonl"
    save_opinion_records([_opinion()], body, manifest)
    old_body = body.read_text()
    old_manifest = manifest.read_text()
    new_record = _opinion(point_id="point-2", state_id="state-2")

    original_replace = opinions_module.os.replace
    calls = 0

    def fail_second_replace(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected manifest rename failure")
        return original_replace(source, destination)

    monkeypatch.setattr(opinions_module.os, "replace", fail_second_replace)
    with pytest.raises(OSError, match="injected"):
        save_opinion_records([new_record], body, manifest, append=True)
    assert body.read_text() == old_body
    assert manifest.read_text() == old_manifest
    assert_opinion_manifest_integrity(body, manifest)


def test_rater_provenance_and_credentials_are_fail_closed():
    llm = _opinion(
        rater_id="llm-seed-7",
        rater_type=RATER_LLM,
        provider="openrouter",
        model="vendor/model-v1",
    )
    llm.validate(_point())
    with pytest.raises(OpinionLabelError, match="LLM rater requires model"):
        replace(llm, model=None).validate()
    with pytest.raises(OpinionLabelError, match="HUMAN rater must not claim a model"):
        replace(_opinion(), model="not-a-human").validate()
    with pytest.raises(OpinionLabelError, match="credential-like material"):
        replace(
            llm,
            rationale="authorization: Bearer this-is-a-secret-token",
        ).validate()

    second = _opinion(point_id="point-2", state_id="state-2")
    with pytest.raises(OpinionLabelError, match="inconsistent provenance"):
        build_point_rater_matrix([_opinion(), replace(second, provider="another-panel")])


def test_opinion_wire_format_cannot_enter_formal_ground_truth_loaders():
    row = _opinion().to_dict()
    assert "effect_status" not in row
    assert "recovery_status" not in row
    assert "normative_risk" not in row
    with pytest.raises(GroundingValidationError):
        GroundingPoint.from_dict(row, validate=True)
    with pytest.raises(EvaluationTruthError):
        EvaluationTruthRecord.from_dict(row)


def test_reserved_formal_artifact_names_are_rejected(tmp_path):
    with pytest.raises(OpinionLabelError, match="formal truth/grounding filename"):
        save_opinion_records(
            [_opinion()],
            tmp_path / "probe_points.jsonl",
            tmp_path / "opinions.manifest.jsonl",
        )


def test_external_batch_import_is_offline_strict_and_point_joined(tmp_path):
    point = _point()
    raw_response = json.dumps({
        "effect": PERCEIVED_CHANGE,
        "recovery": PERCEIVED_RECOVERABLE,
        "normative_risk": NOT_RISKY,
        "confidence": 0.7,
        "rationale": "Imported result; no model call occurs here.",
    })
    payload = {
        "schema_version": OPINION_IMPORT_SCHEMA_VERSION,
        "instrument_id": "iris-opinion-instrument",
        "instrument_version": "v1",
        "prompt_generation_fp": "instrument-prompt-fp-v1",
        "collection_timestamp": "2026-07-14T00:00:00+00:00",
        "import_batch_id": "external-batch-1",
        "code_version": "deadbeef",
        "ratings": [
                {
                    "probe_point_id": "point-1",
                    "state_id": "state-1",
                    "evaluation_case_id": "case-1",
                    "variant": "constraint",
                    "goal_sha256": "1" * 64,
                    "opinion_input_sha256": "b" * 64,
                    "input_messages_sha256": "d" * 64,
                    "raw_response": raw_response,
                    "raw_response_sha256": __import__("hashlib").sha256(
                        raw_response.encode()).hexdigest(),
                    "provider_response_id": "provider-response-1",
                    "response_model": "vendor/model-v1",
                    "finish_reason": "stop",
                "rater_id": "llm-seed-7",
                "rater_type": RATER_LLM,
                "provider": "openrouter",
                "model": "vendor/model-v1",
                "perceived_effect": PERCEIVED_CHANGE,
                "perceived_recoverability": PERCEIVED_RECOVERABLE,
                "normative_risk_opinion": NOT_RISKY,
                "confidence": 0.7,
                "rationale": "Imported result; no model call occurs here.",
                "source_record_id": "provider-output-sha256-1",
            }
        ],
    }
    source = tmp_path / "external-opinions.json"
    source.write_text(json.dumps(payload))
    records = import_external_opinion_ratings(
        source, points={point.probe_point_id: point}
    )
    assert len(records) == 1
    assert records[0].provider == "openrouter"
    assert records[0].model == "vendor/model-v1"

    canonical = tmp_path / "canonical.jsonl"
    canonical.write_text(json.dumps(records[0].to_dict()) + "\n")
    assert import_opinion_records(canonical, points={"point-1": point}) == records

    payload["ratings"][0]["api_key"] = "credential-must-not-enter-artifact"
    with pytest.raises(OpinionLabelError):
        import_external_opinion_ratings(payload, points={"point-1": point})


def test_point_rater_matrix_has_raw_counts_missing_cells_and_unknown_policy():
    records = [
        _opinion(),
        _opinion(
            rater_id="llm-seed-7",
            rater_type=RATER_LLM,
            provider="openrouter",
            model="vendor/model-v1",
            effect="UNKNOWN",
            recovery="UNKNOWN",
            risk="UNKNOWN",
        ),
        _opinion(point_id="point-2", state_id="state-2"),
    ]
    matrix = build_point_rater_matrix(records, unknown_policy="separate")
    assert matrix["matrix_kind"] == "evaluation_case_x_rater"
    assert matrix["raw_counts"]["n_records"] == 3
    assert matrix["raw_counts"]["n_missing_cells"] == 1
    assert matrix["raw_counts"]["perceived_effect"]["raw_counts"] == {
        category: (1 if category == "UNKNOWN" else 2 if category == PERCEIVED_CHANGE else 0)
        for category in PERCEIVED_EFFECTS
    }
    excluded = build_point_rater_matrix(records, unknown_policy="exclude")
    assert "UNKNOWN" not in excluded["raw_counts"]["perceived_effect"][
        "reported_counts"
    ]
    assert excluded["raw_counts"]["perceived_effect"]["unknown_count"] == 1
    with pytest.raises(OpinionLabelError, match="UNKNOWN"):
        build_point_rater_matrix(records, unknown_policy="error")


def test_behavior_grounding_disagreement_keeps_normative_axis_separate():
    point_1 = _point()
    point_2 = _point("point-2", "state-2")
    records = [
        _opinion(),
        _opinion(
            point_id="point-2",
            state_id="state-2",
            effect=PERCEIVED_NO_CHANGE,
            recovery=PERCEIVED_NOT_RECOVERABLE,
            risk=RISKY,
        ),
    ]
    report = build_behavior_grounding_disagreement_matrix(
        records,
        {"point-1": point_1, "point-2": point_2},
        unknown_policy="separate",
    )
    assert report["effect"]["n_total"] == 2
    assert report["effect"]["n_disagreements"] == 1
    assert report["effect"]["raw_matrix"][PERCEIVED_NO_CHANGE][EFFECT_CHANGED] == 1
    assert report["recovery"]["n_disagreements"] == 1
    assert report["recovery"]["raw_matrix"][PERCEIVED_NOT_RECOVERABLE][
        RECOVERY_RECOVERED
    ] == 1
    assert report["normative_vs_behavior_comparison_performed"] is False
    assert report["normative_opinion"]["raw_counts"] == {
        RISKY: 1,
        NOT_RISKY: 1,
        "UNKNOWN": 0,
    }

    unknown = _opinion(
        point_id="point-2",
        state_id="state-2",
        effect="UNKNOWN",
        recovery="UNKNOWN",
        risk=RISKY,
    )
    with pytest.raises(OpinionLabelError, match="UNKNOWN pair"):
        build_behavior_grounding_disagreement_matrix(
            [records[0], unknown],
            {"point-1": point_1, "point-2": point_2},
            unknown_policy="error",
        )
