"""Shared helpers for the Hugging Face Hub upload scripts.

Deliberately light: standard library plus ``huggingface_hub`` only, so the
upload entry points can import these without pulling in torch via utils.
"""

import hashlib
import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path

from huggingface_hub import HfApi


def file_sha256(path: Path) -> str:
    """Streaming SHA-256 of a file."""
    hasher = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def resolve_hf_token(hf_login: bool = False) -> str | None:
    """Resolve the Hub token from the environment, optionally logging in.

    Reads ``HF_TOKEN_ORG`` then ``HF_TOKEN``. When ``hf_login`` is set, performs
    an interactive login with the token and returns ``None`` (the session is
    authenticated, so a per-call token is no longer needed).
    """
    token = os.environ.get("HF_TOKEN_ORG") or os.environ.get("HF_TOKEN") or None
    if hf_login:
        from huggingface_hub import login

        login(token=token)
        return None
    return token


def make_staging_dir(keep_staging_dir: Path | None) -> tuple[Path, Callable[[], None]]:
    """Return a staging directory and a cleanup callback.

    When ``keep_staging_dir`` is given, that directory is (re)created and kept
    for inspection; cleanup is a no-op. Otherwise a temporary directory is used
    and cleanup removes it.
    """
    if keep_staging_dir is not None:
        tmp = Path(keep_staging_dir)
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True, exist_ok=True)
        return tmp, lambda: None

    temp_dir = tempfile.TemporaryDirectory()
    return Path(temp_dir.name), temp_dir.cleanup


def push_folder_to_hub(
    folder: Path,
    repo_id: str,
    *,
    repo_type: str,
    private: bool,
    commit_message: str,
    token: str | None = None,
    api: HfApi | None = None,
) -> HfApi:
    """Create (if needed) and upload a staged folder to a Hub repo."""
    if api is None:
        api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=commit_message,
    )
    return api
