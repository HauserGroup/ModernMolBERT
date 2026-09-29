from pathlib import Path

import pytest

from modernmolbert.common.paths import find_project_root


def test_find_project_root_from_repo_subdirectory(tmp_path: Path):
    root = tmp_path / "repo"
    subdir = root / "examples" / "notebooks"
    subdir.mkdir(parents=True)

    (root / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")

    assert find_project_root(subdir) == root


def test_find_project_root_raises_without_marker(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        find_project_root(tmp_path)
