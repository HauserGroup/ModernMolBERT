import os
from pathlib import Path


_PROJECT_MARKERS = (
    "pyproject.toml",
    ".git",
)


def find_project_root(
    start: str | Path | None = None,
    *,
    markers: tuple[str, ...] = _PROJECT_MARKERS,
) -> Path:
    """Find the repository/project root by walking upward from a start path."""

    env_root = os.environ.get("MODERNMOLBERT_ROOT")
    if env_root:
        root = Path(env_root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"MODERNMOLBERT_ROOT points to a missing path: {root}")
        return root

    if start is None:
        current = Path.cwd()
    else:
        current = Path(start).expanduser().resolve()

    if current.is_file():
        current = current.parent

    current = current.resolve()

    for path in (current, *current.parents):
        if any((path / marker).exists() for marker in markers):
            return path

    marker_text = ", ".join(markers)
    raise FileNotFoundError(
        f"Could not find project root from {current}. Looked for one of: {marker_text}"
    )


def project_path(*parts: str | os.PathLike[str], start: str | Path | None = None) -> Path:
    """Return an absolute path inside the project root."""

    return find_project_root(start=start).joinpath(*parts)
