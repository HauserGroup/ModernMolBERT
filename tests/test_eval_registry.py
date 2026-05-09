import json
from pathlib import Path

import pytest

from modernmolbert.eval.featurizers.dummy import DummyFeaturizer
from modernmolbert.eval.featurizers.rdkit_ecfp import ECFP4Featurizer
from modernmolbert.eval.registry import (
    load_featurizer_config,
    make_featurizer_from_config,
)


def test_make_featurizer_from_dict_keeps_instance_name() -> None:
    featurizer = make_featurizer_from_config(
        {
            "type": "ecfp4",
            "name": "ecfp4_128",
            "n_bits": 128,
            "radius": 2,
        }
    )

    assert isinstance(featurizer, ECFP4Featurizer)
    assert featurizer.name == "ecfp4_128"
    assert featurizer.n_bits == 128
    assert featurizer.radius == 2


def test_make_featurizer_from_json_file(tmp_path: Path) -> None:
    path = tmp_path / "dummy.json"
    path.write_text(
        json.dumps(
            {
                "type": "dummy",
                "name": "dummy_4",
                "n_features": 4,
            }
        ),
        encoding="utf-8",
    )

    featurizer = make_featurizer_from_config(path)

    assert isinstance(featurizer, DummyFeaturizer)
    assert featurizer.name == "dummy_4"
    assert featurizer.n_features == 4


def test_make_featurizer_from_config_requires_type() -> None:
    with pytest.raises(ValueError, match="type"):
        make_featurizer_from_config(
            {
                "name": "ecfp4_128",
                "n_bits": 128,
            }
        )


def test_load_featurizer_config_rejects_non_json_file(tmp_path: Path) -> None:
    path = tmp_path / "dummy.yaml"
    path.write_text("type: dummy\n", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON"):
        load_featurizer_config(path)


def test_load_featurizer_config_rejects_non_object_json(tmp_path: Path) -> None:
    path = tmp_path / "dummy.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON object"):
        load_featurizer_config(path)
