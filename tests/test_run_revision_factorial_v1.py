import pytest

from scripts import run_revision_factorial_v1 as launcher


def test_launcher_rejects_occupied_gpu(monkeypatch):
    monkeypatch.setattr(
        launcher.subprocess,
        "check_output",
        lambda *_args, **_kwargs: "106138, /opt/lab/envs/esmfold2-py312/bin/python\n",
    )
    with pytest.raises(RuntimeError, match="already in use"):
        launcher.check_gpu_available()


def test_launcher_accepts_idle_gpu(monkeypatch):
    monkeypatch.setattr(launcher.subprocess, "check_output", lambda *_args, **_kwargs: "")
    launcher.check_gpu_available()
