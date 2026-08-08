"""Keychain write-back on Claude Code credential refresh.

Claude Code's refresh tokens are single-use — refreshing rotates the pair and
invalidates the old token. On macOS Claude Code reads its credentials from the
Keychain, so a refresh that only rewrote ~/.claude/.credentials.json left the
Keychain holding an invalidated refresh token: Claude Code's next refresh then
failed with invalid_grant and forced the user to log in again.

Every test here mocks ``subprocess.run``; none touches a real Keychain.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from agent.anthropic_adapter import (
    _CLAUDE_CODE_KEYCHAIN_SERVICE,
    _write_claude_code_credentials,
    _write_claude_code_credentials_to_keychain,
)

# These tests drive the Keychain helpers directly with explicit platform and
# subprocess mocks, so they opt out of the suite-wide neutralizer.
pytestmark = pytest.mark.allow_macos_keychain

EXISTING_PAYLOAD = {
    "claudeAiOauth": {
        "accessToken": "old-access",
        "refreshToken": "old-refresh",
        "expiresAt": 1000,
        "scopes": ["user:inference", "user:profile"],
        "subscriptionType": "max",
    }
}
ATTRIBUTES_OUTPUT = 'keychain: "login.keychain-db"\n    "acct"<blob>="alice"\n    "svce"<blob>="Claude Code-credentials"\n'

NEW_OAUTH = {
    "accessToken": "new-access",
    "refreshToken": "new-refresh",
    "expiresAt": 2000,
}


def _security_stub(*, entry_exists=True, calls=None):
    """Stand in for the `security` CLI reads."""

    def _run(cmd, *args, **kwargs):
        if calls is not None:
            calls.append((cmd, kwargs))
        if cmd[1] == "find-generic-password":
            if not entry_exists:
                return MagicMock(returncode=1, stdout="", stderr="")
            if "-w" in cmd:
                return MagicMock(returncode=0, stdout=json.dumps(EXISTING_PAYLOAD))
            return MagicMock(returncode=0, stdout=ATTRIBUTES_OUTPUT)
        return MagicMock(returncode=0 if write_ok else 1, stdout="", stderr="")

    return _run


class TestKeychainWriteBack:
    def test_updates_the_entry_claude_code_reads(self):
        calls = []
        with patch("agent.anthropic_adapter.platform.system", return_value="Darwin"), \
             patch("agent.anthropic_adapter.subprocess.run",
                   side_effect=_security_stub(calls=calls)), \
             patch(
                 "agent.anthropic_adapter._update_macos_keychain_secret",
                 return_value=True,
             ) as native_update:
            assert _write_claude_code_credentials_to_keychain(NEW_OAUTH) is True

        service, account, secret = native_update.call_args.args
        assert service == _CLAUDE_CODE_KEYCHAIN_SERVICE
        assert account == "alice"
        written = json.loads(secret)
        assert written["claudeAiOauth"]["accessToken"] == "new-access"
        assert written["claudeAiOauth"]["refreshToken"] == "new-refresh"

    def test_targets_the_existing_account(self):
        """security keys on (service, account); the wrong account forks a duplicate."""
        calls = []
        with patch("agent.anthropic_adapter.platform.system", return_value="Darwin"), \
             patch("agent.anthropic_adapter.subprocess.run",
                   side_effect=_security_stub(calls=calls)), \
             patch(
                 "agent.anthropic_adapter._update_macos_keychain_secret",
                 return_value=True,
             ) as native_update:
            _write_claude_code_credentials_to_keychain(NEW_OAUTH)

        assert native_update.call_args.args[1] == "alice"

    def test_secret_never_reaches_the_argument_list(self):
        """`ps` is world-readable; the token must travel on stdin only."""
        calls = []
        with patch("agent.anthropic_adapter.platform.system", return_value="Darwin"), \
             patch("agent.anthropic_adapter.subprocess.run",
                   side_effect=_security_stub(calls=calls)), \
             patch(
                 "agent.anthropic_adapter._update_macos_keychain_secret",
                 return_value=True,
             ) as native_update:
            _write_claude_code_credentials_to_keychain(NEW_OAUTH)

        assert not any(
            "new-refresh" in part or "new-access" in part
            for cmd, _ in calls
            for part in cmd
        )
        assert b"new-refresh" in native_update.call_args.args[2]

    def test_long_payload_is_not_sent_through_the_128_byte_cli_prompt(self):
        """The real `security -w` prompt truncates long secrets at 128 bytes."""
        calls = []
        long_oauth = dict(NEW_OAUTH, accessToken="a" * 256)
        with patch("agent.anthropic_adapter.platform.system", return_value="Darwin"), \
             patch("agent.anthropic_adapter.subprocess.run",
                   side_effect=_security_stub(calls=calls)), \
             patch(
                 "agent.anthropic_adapter._update_macos_keychain_secret",
                 return_value=True,
             ) as native_update:
            _write_claude_code_credentials_to_keychain(long_oauth)

        secret = native_update.call_args.args[2]
        assert len(secret) > 128
        assert json.loads(secret)["claudeAiOauth"]["accessToken"] == "a" * 256
        assert not any(cmd[1] == "add-generic-password" for cmd, _ in calls)

    def test_preserves_sibling_fields(self):
        """Dropping scopes/subscriptionType invalidates the credential for Claude Code."""
        calls = []
        with patch("agent.anthropic_adapter.platform.system", return_value="Darwin"), \
             patch("agent.anthropic_adapter.subprocess.run",
                   side_effect=_security_stub(calls=calls)), \
             patch(
                 "agent.anthropic_adapter._update_macos_keychain_secret",
                 return_value=True,
             ) as native_update:
            _write_claude_code_credentials_to_keychain(NEW_OAUTH)

        written = json.loads(native_update.call_args.args[2])["claudeAiOauth"]
        assert written["scopes"] == ["user:inference", "user:profile"]
        assert written["subscriptionType"] == "max"

    def test_never_creates_an_entry_that_does_not_exist(self):
        """Installs that keep credentials in the JSON file must stay untouched."""
        calls = []
        with patch("agent.anthropic_adapter.platform.system", return_value="Darwin"), \
             patch("agent.anthropic_adapter.subprocess.run",
                   side_effect=_security_stub(entry_exists=False, calls=calls)), \
             patch(
                 "agent.anthropic_adapter._update_macos_keychain_secret"
             ) as native_update:
            assert _write_claude_code_credentials_to_keychain(NEW_OAUTH) is False

        native_update.assert_not_called()

    def test_noop_off_darwin(self):
        with patch("agent.anthropic_adapter.platform.system", return_value="Linux"), \
             patch("agent.anthropic_adapter.subprocess.run") as mock_run:
            assert _write_claude_code_credentials_to_keychain(NEW_OAUTH) is False
        mock_run.assert_not_called()

    def test_survives_a_missing_security_binary(self):
        with patch("agent.anthropic_adapter.platform.system", return_value="Darwin"), \
             patch("agent.anthropic_adapter.subprocess.run",
                   side_effect=OSError("security not found")):
            assert _write_claude_code_credentials_to_keychain(NEW_OAUTH) is False

    def test_reports_failure_when_security_exits_nonzero(self):
        with patch("agent.anthropic_adapter.platform.system", return_value="Darwin"), \
             patch("agent.anthropic_adapter.subprocess.run",
                   side_effect=_security_stub()), \
             patch(
                 "agent.anthropic_adapter._update_macos_keychain_secret",
                 return_value=False,
             ):
            assert _write_claude_code_credentials_to_keychain(NEW_OAUTH) is False


class TestRefreshKeepsBothStoresInSync:
    def test_file_write_also_mirrors_to_the_keychain(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)
        with patch(
            "agent.anthropic_adapter._write_claude_code_credentials_to_keychain",
            return_value=True,
        ) as mock_keychain:
            _write_claude_code_credentials("acc", "ref", 4242)

        on_disk = json.loads(
            (tmp_path / ".claude" / ".credentials.json").read_text()
        )["claudeAiOauth"]
        assert on_disk["accessToken"] == "acc"
        assert on_disk["refreshToken"] == "ref"

        mirrored = mock_keychain.call_args[0][0]
        assert mirrored["accessToken"] == "acc"
        assert mirrored["refreshToken"] == "ref"
        assert mirrored["expiresAt"] == 4242

    def test_mirrors_even_when_the_file_write_fails(self, tmp_path, monkeypatch):
        """The Keychain is the store Claude Code reads on macOS — it must still land."""
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)
        with patch("agent.anthropic_adapter.os.open", side_effect=OSError("read-only fs")), \
             patch(
                 "agent.anthropic_adapter._write_claude_code_credentials_to_keychain",
                 return_value=True,
             ) as mock_keychain:
            _write_claude_code_credentials("acc", "ref", 4242)

        mock_keychain.assert_called_once()

    def test_keychain_failure_never_fails_a_persisted_refresh(self, tmp_path, monkeypatch):
        """A locked Keychain must not discard a token pair already on disk."""
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)
        with patch(
            "agent.anthropic_adapter._write_claude_code_credentials_to_keychain",
            side_effect=RuntimeError("keychain locked"),
        ):
            _write_claude_code_credentials("acc", "ref", 4242)

        on_disk = json.loads(
            (tmp_path / ".claude" / ".credentials.json").read_text()
        )["claudeAiOauth"]
        assert on_disk["accessToken"] == "acc"
