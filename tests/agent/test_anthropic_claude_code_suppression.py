"""Removing Hermes' Claude Code credential must actually take effect.

`hermes auth remove anthropic <claude_code entry>` suppresses the source so
Hermes stops reading Claude Code's credential store (the store itself is
deliberately left intact — the user's Claude Code install keeps working).

The credential pool honoured that suppression when re-seeding, but
resolve_anthropic_token() read Claude Code's store directly at step 4, ahead
of the pool, with no suppression check. The removal reported success and
changed nothing, so a user whose Claude Code is logged into account A could
not point Hermes at account B.
"""

from unittest.mock import patch

import pytest

from agent.anthropic_adapter import resolve_anthropic_token

CLAUDE_CODE_TOKEN = "sk-ant-oat01-from-claude-code"
POOL_TOKEN = "sk-ant-oat01-from-hermes-pool"

CLAUDE_CODE_CREDS = {
    "accessToken": CLAUDE_CODE_TOKEN,
    "refreshToken": "refresh-cc",
    "expiresAt": 9_999_999_999_000,
    "source": "macos_keychain",
}


@pytest.fixture(autouse=True)
def _no_env_credentials(monkeypatch):
    """Steps 1-3 read env vars; keep them out of the way."""
    for var in ("ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(
        "agent.anthropic_adapter._getenv", lambda _name, *a, **k: ""
    )


def _resolve(*, suppressed: bool, pool_token=POOL_TOKEN):
    with patch(
        "agent.anthropic_adapter.read_claude_code_credentials",
        return_value=dict(CLAUDE_CODE_CREDS),
    ), patch(
        "agent.anthropic_adapter._resolve_claude_code_token_from_credentials",
        return_value=CLAUDE_CODE_TOKEN,
    ), patch(
        "agent.anthropic_adapter._resolve_anthropic_pool_token",
        return_value=pool_token,
    ), patch(
        "agent.anthropic_adapter._claude_code_source_suppressed",
        return_value=suppressed,
    ):
        return resolve_anthropic_token()


class TestClaudeCodeSuppression:
    def test_claude_code_wins_when_not_suppressed(self):
        """Default behaviour is unchanged: step 4 still precedes the pool."""
        assert _resolve(suppressed=False) == CLAUDE_CODE_TOKEN

    def test_pool_wins_once_the_user_removed_claude_code(self):
        """The whole point of the removal: Hermes switches accounts."""
        assert _resolve(suppressed=True) == POOL_TOKEN

    def test_suppression_does_not_invent_a_token(self):
        """With nothing else configured, resolution fails rather than falling back."""
        assert _resolve(suppressed=True, pool_token=None) is None

    def test_claude_code_store_is_never_read_when_suppressed(self):
        """Reading it would defeat the removal even if the value went unused."""
        with patch(
            "agent.anthropic_adapter.read_claude_code_credentials"
        ) as mock_read, patch(
            "agent.anthropic_adapter._resolve_claude_code_token_from_credentials"
        ) as mock_resolve_cc, patch(
            "agent.anthropic_adapter._resolve_anthropic_pool_token",
            return_value=POOL_TOKEN,
        ), patch(
            "agent.anthropic_adapter._claude_code_source_suppressed",
            return_value=True,
        ):
            assert resolve_anthropic_token() == POOL_TOKEN

        mock_read.assert_not_called()
        mock_resolve_cc.assert_not_called()


class TestSuppressionLookup:
    def test_reads_the_suppression_recorded_by_auth_remove(self, tmp_path, monkeypatch):
        """End-to-end against the real auth store, no mocking of the lookup."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from hermes_cli.auth import suppress_credential_source
        from agent.anthropic_adapter import _claude_code_source_suppressed

        assert _claude_code_source_suppressed() is False
        suppress_credential_source("anthropic", "claude_code")
        assert _claude_code_source_suppressed() is True

    def test_other_providers_are_unaffected(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from hermes_cli.auth import suppress_credential_source
        from agent.anthropic_adapter import _claude_code_source_suppressed

        suppress_credential_source("openai-codex", "claude_code")
        assert _claude_code_source_suppressed() is False

    def test_fails_open_when_the_auth_store_cannot_be_read(self):
        """An unreadable store must not strip a working credential source."""
        from agent.anthropic_adapter import _claude_code_source_suppressed

        with patch(
            "hermes_cli.auth.is_source_suppressed",
            side_effect=OSError("auth.json unreadable"),
        ):
            assert _claude_code_source_suppressed() is False
