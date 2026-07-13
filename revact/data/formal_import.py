"""Fail-closed import boundary for externally measured formal artifacts.

This module never invents provenance, labels, or normative truth.  It only
admits complete, versioned JSONL records through the canonical validators and
materializes the paired integrity manifests.  Live probe runners and policy
truth authoring tools may therefore evolve independently without bypassing the
same publication gate used by assembly and training.
"""
from __future__ import annotations

from pathlib import Path

from .candidates import (CandidateValidationError,
                         assert_candidate_manifest_integrity,
                         validate_candidate_snapshot_artifact)
from .governance import point_candidate_reasons
from ..eval.truth import (EvaluationTruthError, load_truth_records,
                          save_truth_records)
from ..grounding.schema import (GroundingValidationError,
                                assert_manifest_integrity,
                                load_probe_points, save_probe_points)


class FormalImportError(ValueError):
    """Raised when an import is empty, self-referential, or invalid."""


def import_grounding_points(source: Path, data_root: Path) -> dict:
    """Append validated point records without mutating ``source``.

    A point body is sufficient as input because the canonical manifest is
    regenerated from the validated record bytes.  The destination remains
    immutable by ``probe_point_id``.
    """
    source = Path(source).resolve()
    data_root = Path(data_root).resolve()
    body = data_root / "grounded" / "probe_points.jsonl"
    manifest = data_root / "grounded" / "POINT_MANIFEST.jsonl"
    if source == body.resolve():
        raise FormalImportError("source must not be the canonical destination")
    try:
        points = load_probe_points(source, validate=True)
    except (GroundingValidationError, OSError, ValueError) as exc:
        raise FormalImportError(str(exc)) from exc
    if not points:
        raise FormalImportError("grounding import contains zero records")
    candidate_path = data_root / "raw" / "candidates" / "iris_candidates.v3.jsonl"
    try:
        candidates = assert_candidate_manifest_integrity(candidate_path)
        validate_candidate_snapshot_artifact(
            candidates, data_root / "raw" / "state_bank")
    except (CandidateValidationError, OSError, ValueError) as exc:
        raise FormalImportError(
            f"candidate artifact invalid: {exc}") from exc
    candidate_problems = [
        f"{point.probe_point_id}: {reason}"
        for point in points.values()
        for reason in point_candidate_reasons(point, candidates)
    ]
    if candidate_problems:
        raise FormalImportError(
            "grounding/candidate join failed: " +
            " | ".join(candidate_problems[:10]))
    before = len(load_probe_points(body, validate=True)) if body.exists() else 0
    try:
        save_probe_points(points.values(), body, manifest, append=True)
        assert_manifest_integrity(body, manifest)
    except (GroundingValidationError, OSError, ValueError) as exc:
        raise FormalImportError(str(exc)) from exc
    return {
        "kind": "grounding_points", "source": str(source),
        "body": str(body), "manifest": str(manifest),
        "n_imported": len(points), "n_before": before,
        "n_after": before + len(points),
    }


def import_evaluation_truth(source: Path, data_root: Path) -> dict:
    """Append authored point×variant policy truth after an exact point join.

    Recoverability fields are checked against the canonical point.  Normative
    fields must be explicitly present in the source; no recovery-to-safety
    conversion exists in this path.
    """
    source = Path(source).resolve()
    data_root = Path(data_root).resolve()
    points_body = data_root / "grounded" / "probe_points.jsonl"
    points_manifest = data_root / "grounded" / "POINT_MANIFEST.jsonl"
    body = data_root / "eval" / "truth.jsonl"
    manifest = data_root / "eval" / "TRUTH_MANIFEST.jsonl"
    if source == body.resolve():
        raise FormalImportError("source must not be the canonical destination")
    try:
        assert_manifest_integrity(points_body, points_manifest)
        points = load_probe_points(points_body, validate=True)
        if not points:
            raise FormalImportError(
                "evaluation truth cannot be imported before formal grounding")
        records = load_truth_records(source, points=points)
    except FormalImportError:
        raise
    except (GroundingValidationError, EvaluationTruthError, OSError,
            ValueError) as exc:
        raise FormalImportError(str(exc)) from exc
    if not records:
        raise FormalImportError("evaluation truth import contains zero records")
    before = len(load_truth_records(body, points=points)) if body.exists() else 0
    try:
        save_truth_records(records.values(), body, manifest, append=True)
    except (EvaluationTruthError, OSError, ValueError) as exc:
        raise FormalImportError(str(exc)) from exc
    return {
        "kind": "evaluation_truth", "source": str(source),
        "body": str(body), "manifest": str(manifest),
        "n_imported": len(records), "n_before": before,
        "n_after": before + len(records),
    }
