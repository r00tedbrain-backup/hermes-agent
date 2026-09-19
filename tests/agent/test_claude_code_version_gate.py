"""Regression tests for the reported Claude Code version (the model gate).

Anthropic gates model access on the version we report in the OAuth user-agent,
server-side. Reporting one below the gate returns HTTP 400:

    Claude Code 2.1.74 does not support this model;
    version 2.1.251 or newer is required.

Two independent paths produced a below-gate report, both surfacing as that 400:

* The Electron desktop app launches with the bare macOS PATH
  ``/usr/bin:/bin:/usr/sbin:/sbin``, so a PATH-only lookup for the CLI found
  nothing even with it installed, and detection fell through to the constant.
* A user whose installed CLI predates the gate reported that older version.

The invariant that closes both: never report below the fallback constant.
"""

from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest

from agent.anthropic_adapter import (
    _CLAUDE_CODE_VERSION_FALLBACK,
    _detect_claude_code_version,
    _version_tuple,
)

# The gate that motivated this module. A regression here is a user-visible 400,
# so it is asserted against the constant rather than trusting it by inspection.
GATE = (2, 1, 251)

BARE_GUI_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"


def _fake_cli(version: str, *, found_at: str):
    """Patch context where the only CLI on the box reports ``version``."""
    completed = subprocess.CompletedProcess(
        args=[found_at, "--version"], returncode=0, stdout=f"{version} (Claude Code)", stderr=""
    )
    return (
        patch("shutil.which", lambda name: found_at if name == "claude" else None),
        patch("os.path.isfile", lambda path: path == found_at),
        patch("subprocess.run", lambda *a, **k: completed),
    )


class TestVersionTuple:
    def test_orders_numerically_not_lexically(self):
        """``2.1.99`` sorts after ``2.1.258`` as a string but is the older release."""
        assert _version_tuple("2.1.99") < _version_tuple("2.1.258")

    @pytest.mark.parametrize("malformed", ["", "2.x.1", "latest", "2.1.258-beta"])
    def test_malformed_is_falsy_and_never_raises(self, malformed):
        assert _version_tuple(malformed) == ()


class TestReportedVersionClearsGate:
    def test_fallback_constant_clears_the_gate(self):
        """The floor must satisfy the gate on its own — it is the no-CLI answer."""
        assert _version_tuple(_CLAUDE_CODE_VERSION_FALLBACK) >= GATE

    def test_no_cli_anywhere_reports_the_floor(self):
        """Detection failure must not report a below-gate version."""
        with patch("shutil.which", return_value=None), patch(
            "subprocess.run", side_effect=FileNotFoundError
        ):
            assert _detect_claude_code_version() == _CLAUDE_CODE_VERSION_FALLBACK

    def test_bare_gui_path_still_finds_cli_in_user_prefix(self, monkeypatch):
        """The desktop-app bug: CLI in ~/.local/bin, PATH lacking it.

        ``shutil.which`` returns None under the bare GUI PATH, so only the
        explicit prefix probe can find the CLI here.
        """
        monkeypatch.setenv("PATH", BARE_GUI_PATH)
        home = "/Users/tester"
        monkeypatch.setattr("os.path.expanduser", lambda p: p.replace("~", home, 1))
        installed = f"{home}/.local/bin/claude"
        newer = "2.1.999"

        with patch("shutil.which", return_value=None), patch(
            "os.path.isfile", lambda path: path == installed
        ), patch(
            "subprocess.run",
            lambda *a, **k: subprocess.CompletedProcess(
                args=a[0], returncode=0, stdout=f"{newer} (Claude Code)", stderr=""
            ),
        ):
            assert _detect_claude_code_version() == newer

    def test_installed_cli_older_than_floor_is_ignored(self):
        """A stale local CLI must not drag the reported version below the gate."""
        whicher, isfiler, runner = _fake_cli("2.1.100", found_at="/usr/local/bin/claude")
        with whicher, isfiler, runner:
            assert _detect_claude_code_version() == _CLAUDE_CODE_VERSION_FALLBACK

    def test_installed_cli_newer_than_floor_is_preferred(self):
        """Users who update the CLI get the newer version without a Hermes release."""
        newer = "2.9.9"
        whicher, isfiler, runner = _fake_cli(newer, found_at="/usr/local/bin/claude")
        with whicher, isfiler, runner:
            assert _detect_claude_code_version() == newer

    def test_nonzero_exit_falls_back(self):
        """A CLI that errors out must not yield a garbage version string."""
        failed = subprocess.CompletedProcess(
            args=["claude", "--version"], returncode=1, stdout="", stderr="boom"
        )
        with patch("shutil.which", return_value="/usr/local/bin/claude"), patch(
            "os.path.isfile", return_value=True
        ), patch("subprocess.run", lambda *a, **k: failed):
            assert _detect_claude_code_version() == _CLAUDE_CODE_VERSION_FALLBACK
