"""Build tools must be discoverable when a venv's Python is run directly."""

import shutil
import subprocess
import sys

import pytest

from scripts.iris.qualify_cuda import prepare_build_tools


def test_venv_ninja_is_visible_to_bare_name_child_processes(tmp_path, monkeypatch):
    scripts = tmp_path / "venv/bin"
    scripts.mkdir(parents=True)
    ninja = scripts / "ninja"
    ninja.write_text("#!/bin/sh\nprintf '1.13.0\\n'\n")
    ninja.chmod(0o755)
    monkeypatch.setattr(sys, "executable", str(scripts / "python"))
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert shutil.which("ninja") is None
    with pytest.raises(FileNotFoundError):
        subprocess.check_output(["ninja", "--version"], text=True)
    info = prepare_build_tools()
    assert info == {"ninja": str(ninja), "ninja_version": "1.13.0"}
    # This is the lookup FlashInfer's subprocess uses, not an absolute path.
    assert subprocess.check_output(["ninja", "--version"], text=True).strip() == "1.13.0"


def test_missing_ninja_reports_the_active_python_install_command(tmp_path, monkeypatch):
    python = tmp_path / "venv/bin/python"
    monkeypatch.setattr(sys, "executable", str(python))
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    with pytest.raises(RuntimeError, match="ninja is not installed or executable") as exc:
        prepare_build_tools()
    assert f"uv pip install --python {python} ninja" in str(exc.value)
