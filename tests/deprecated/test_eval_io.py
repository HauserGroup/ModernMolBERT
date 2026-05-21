import math

import numpy as np
import pytest

from modernmolbert.eval.io import (
    ensure_dir,
    hash_strings,
    load_numpy,
    read_json,
    save_numpy,
    write_json,
)


def test_ensure_dir_creates_nested_directory_and_returns_path(tmp_path) -> None:
    nested = tmp_path / "one" / "two"

    created = ensure_dir(nested)

    assert created == nested
    assert nested.is_dir()


def test_json_helpers_round_trip_nested_data(tmp_path) -> None:
    path = tmp_path / "result.json"
    data = {
        "name": "toy",
        "metrics": {"loss": 1.25, "maybe_nan": math.nan},
        "rows": [1, 2, None],
    }

    write_json(path, data)

    loaded = read_json(path)
    assert loaded["name"] == "toy"
    assert loaded["rows"] == [1, 2, None]
    assert math.isnan(loaded["metrics"]["maybe_nan"])
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_hash_strings_is_ordered_and_delimited() -> None:
    assert hash_strings(["ab", "c"]) != hash_strings(["a", "bc"])
    assert hash_strings(["a", "b"]) != hash_strings(["b", "a"])
    assert hash_strings(["a", "b"]) == hash_strings(["a", "b"])


def test_numpy_helpers_round_trip_without_pickle(tmp_path) -> None:
    path = tmp_path / "features.npy"
    arr = np.array([[1.0, 2.0], [3.5, 4.5]], dtype=np.float32)

    save_numpy(path, arr)

    loaded = load_numpy(path)
    np.testing.assert_array_equal(loaded, arr)
    assert loaded.dtype == np.float32


def test_load_numpy_rejects_pickled_object_arrays(tmp_path) -> None:
    path = tmp_path / "objects.npy"
    np.save(path, np.array([{"not": "safe"}], dtype=object))

    with pytest.raises(ValueError, match="Object arrays cannot be loaded"):
        load_numpy(path)
