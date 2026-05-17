from pathlib import Path

import pytest

from modernmolbert.common.paths import find_project_root, project_path


def test_find_project_root_from_repo_subdirectory(tmp_path: Path):
    root = tmp_path / "repo"
    subdir = root / "examples" / "notebooks"
    subdir.mkdir(parents=True)

    (root / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")

    assert find_project_root(subdir) == root


def test_find_project_root_from_file_path(tmp_path: Path):
    root = tmp_path / "repo"
    subdir = root / "examples"
    subdir.mkdir(parents=True)

    file_path = subdir / "example.py"
    file_path.write_text("# example\n", encoding="utf-8")

    (root / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")

    assert find_project_root(file_path) == root


def test_find_project_root_raises_without_marker(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        find_project_root(tmp_path)


def test_project_path_joins_under_root(tmp_path: Path):
    root = tmp_path / "repo"
    subdir = root / "examples"
    subdir.mkdir(parents=True)

    (root / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")

    assert project_path("data", "x", start=subdir) == root / "data" / "x"


def test_find_project_root_env_override(tmp_path: Path, monkeypatch):
    root = tmp_path / "custom-root"
    root.mkdir()

    monkeypatch.setenv("MODERNMOLBERT_ROOT", str(root))

    assert find_project_root() == root
