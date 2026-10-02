"""The seed-42 CV metadata repair preserves every other manifest field."""

import json

import numpy as np
import pytest
from sklearn.model_selection import KFold, StratifiedKFold, check_cv

from correct_revision_cv_manifest import correct, sha256
from modernmolbert.eval.benchmarking_molecular_models.supervised.const import (
    PRODUCTION_CV_POLICY,
)


def test_recorded_production_cv_matches_sklearn_integer_cv():
    labels = np.tile([0, 1, 0, 1, 1], 6)
    for y, expected_type in (
        (labels, StratifiedKFold),
        (np.column_stack([labels, 1 - labels]), KFold),
    ):
        splitter = check_cv(5, y=y, classifier=True)
        assert isinstance(splitter, expected_type)
        assert splitter.n_splits == PRODUCTION_CV_POLICY["folds"]
        assert splitter.shuffle == PRODUCTION_CV_POLICY["shuffle"]
        assert splitter.random_state == PRODUCTION_CV_POLICY["seed"]


def test_correct_seed42_cv_metadata_only(tmp_path):
    path = tmp_path / "evaluation_manifest.json"
    record = {
        "schema": 2,
        "seed": 42,
        "run_ids": [f"run{i}" for i in range(5)],
        "tasks": {f"task{i}": {"digest": str(i)} for i in range(25)},
        "missing_labels": "as-negative",
        "cv": {"folds": 5, "shuffle": True, "seed": 0},
        "campaign_manifest_sha256": "a" * 64,
    }
    path.write_text(json.dumps(record), encoding="utf-8")
    original = path.read_bytes()
    with pytest.raises(ValueError, match="Manifest hash differs"):
        correct(path, "0" * 64)
    old_hash, new_hash = correct(path, sha256(original))
    assert old_hash == sha256(original)
    assert new_hash == sha256(path.read_bytes())
    assert (tmp_path / "evaluation_manifest.cv_metadata_original.json").read_bytes() == original
    updated = json.loads(path.read_text(encoding="utf-8"))
    assert updated.pop("cv") == {"folds": 5, "shuffle": False, "seed": None}
    record.pop("cv")
    assert updated == record
    with pytest.raises(ValueError, match="Manifest hash differs"):
        correct(path, old_hash)
