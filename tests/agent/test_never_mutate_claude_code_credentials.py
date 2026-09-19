"""Hermes must never mutate Claude Code's credentials.

Anthropic's OAuth refresh tokens are single-use: spending one rotates the pair
and invalidates the copy the other client still holds. Hermes used to refresh
Claude Code's credentials and write the rotated pair back into Claude Code's
own stores (~/.claude/.credentials.json and the macOS Keychain). Both halves
caused real breakage — a truncated Keychain write corrupted the entry, and a
divergence between the two stores left Claude Code with a refresh token the
server had already invalidated, forcing a re-login.

The contract is now read-only: use Claude Code's token while it is valid,
adopt one Claude Code refreshed itself, and otherwise fall through to a
Hermes-owned credential. Never write, never rotate.
"""

from unittest.mock import patch

import pytest

from agent.anthropic_adapter import _refresh_oauth_token
from agent.credential_pool import (
    AUTH_TYPE_OAUTH,
    STATUS_OK,
    CredentialPool,
    PooledCredential,
)

EXPIRED_CC_CREDS = {
    "accessToken": "cc-expired",
    "refreshToken": "cc-refresh",
    "expiresAt": 1,
}


class TestResolverNeverRotates:
    def test_expired_claude_code_token_is_not_refreshed(self, monkeypatch):
        monkeypatch.setattr(
            "agent.anthropic_adapter.read_claude_code_credentials",
            lambda: dict(EXPIRED_CC_CREDS),
        )
        with patch(
            "agent.anthropic_adapter.refresh_anthropic_oauth_pure"
        ) as mock_refresh:
            assert _refresh_oauth_token(dict(EXPIRED_CC_CREDS)) is None
        mock_refresh.assert_not_called()

    def test_a_token_claude_code_refreshed_itself_is_still_adopted(self, monkeypatch):
        """Read-only does not mean ignoring Claude Code — only never writing it."""
        fresh = {
            "accessToken": "cc-fresh",
            "refreshToken": "cc-refresh-2",
            "expiresAt": 9_999_999_999_000,
        }
        monkeypatch.setattr(
            "agent.anthropic_adapter.read_claude_code_credentials", lambda: fresh
        )
        with patch(
            "agent.anthropic_adapter.refresh_anthropic_oauth_pure"
        ) as mock_refresh:
            assert _refresh_oauth_token(dict(EXPIRED_CC_CREDS)) == "cc-fresh"
        mock_refresh.assert_not_called()


class TestPoolNeverRotatesClaudeCodeEntries:
    def _pool(self, tmp_path, monkeypatch, source):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        entry = PooledCredential(
            provider="anthropic",
            id="abc123",
            label="entry",
            auth_type=AUTH_TYPE_OAUTH,
            priority=0,
            source=source,
            access_token="stale",
            refresh_token="pool-refresh",
            expires_at_ms=1,
            last_status=STATUS_OK,
        )
        return CredentialPool("anthropic", [entry]), entry

    def test_claude_code_sourced_entry_is_left_alone(self, tmp_path, monkeypatch):
        pool, entry = self._pool(tmp_path, monkeypatch, "claude_code")
        with patch(
            "agent.anthropic_adapter.refresh_anthropic_oauth_pure"
        ) as mock_refresh:
            assert pool._refresh_entry_impl(entry, force=True) is None
        mock_refresh.assert_not_called()

    def test_hermes_owned_entry_still_refreshes(self, tmp_path, monkeypatch):
        """The restriction is about ownership, not about Anthropic in general."""
        pool, entry = self._pool(tmp_path, monkeypatch, "manual:hermes_pkce")
        with patch(
            "agent.anthropic_adapter.refresh_anthropic_oauth_pure",
            return_value={
                "access_token": "new",
                "refresh_token": "new-refresh",
                "expires_at_ms": 9_999_999_999_000,
            },
        ) as mock_refresh:
            updated = pool._refresh_entry_impl(entry, force=True)
        mock_refresh.assert_called_once()
        assert updated is not None and updated.access_token == "new"


class TestNoWriterRemains:
    def test_the_credential_writers_are_gone(self):
        """A writer with no callers is a loaded gun — it must not exist."""
        import agent.anthropic_adapter as adapter

        for removed in (
            "_write_claude_code_credentials",
            "_write_claude_code_credentials_to_keychain",
            "_update_macos_keychain_secret",
        ):
            assert not hasattr(adapter, removed), f"{removed} is back"

    def test_reading_claude_code_still_works(self):
        """Removing the writers must not take the readers with them."""
        import agent.anthropic_adapter as adapter

        assert hasattr(adapter, "read_claude_code_credentials")
        assert hasattr(adapter, "_read_claude_code_credentials_from_keychain")
