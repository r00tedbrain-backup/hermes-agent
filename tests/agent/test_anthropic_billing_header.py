"""Claude Code billing header on Anthropic OAuth requests.

Background: Anthropic's subscription classifier only bills a Messages request
to the plan's included quota when it carries Claude Code's
``x-anthropic-billing-header:`` block as the first system block. Without it
the request is routed to the "extra usage" lane and — on an account with no
extra-usage credits — rejected with an HTTP 400 whose message reads like a
billing problem ("You're out of extra usage") even though the same account is
serving Claude Code fine at that moment.

These tests pin the behaviour that fix depends on, not the reverse-engineered
digest values themselves (those are Anthropic's to rotate).
"""

import re

import pytest

from agent.anthropic_adapter import (
    _CLAUDE_CODE_SYSTEM_PREFIX,
    build_anthropic_kwargs,
    build_billing_header_value,
)

BILLING_PREFIX = "x-anthropic-billing-header:"
FIXED_VERSION = "2.1.87"


@pytest.fixture(autouse=True)
def _pinned_claude_code_version(monkeypatch):
    """Keep tests hermetic: never shell out to `claude --version`."""
    monkeypatch.setattr(
        "agent.anthropic_adapter._get_claude_code_version", lambda: FIXED_VERSION
    )


def _system_texts(kwargs):
    return [b["text"] for b in kwargs["system"] if isinstance(b, dict)]


def _oauth_kwargs(messages, **overrides):
    params = {
        "model": "claude-sonnet-4-6",
        "messages": messages,
        "tools": None,
        "max_tokens": 1024,
        "reasoning_config": None,
        "is_oauth": True,
    }
    params.update(overrides)
    return build_anthropic_kwargs(**params)


class TestBillingHeaderValue:
    def test_has_the_three_fields_anthropic_routes_on(self):
        value = build_billing_header_value(
            [{"role": "user", "content": "hello world"}], version=FIXED_VERSION
        )
        assert value.startswith(BILLING_PREFIX)
        assert f"cc_version={FIXED_VERSION}." in value
        assert "cc_entrypoint=sdk-cli;" in value
        assert re.search(r"cch=[0-9a-f]{5};", value)

    def test_version_carries_a_three_char_suffix(self):
        value = build_billing_header_value(
            [{"role": "user", "content": "hello world"}], version=FIXED_VERSION
        )
        version_field = re.search(r"cc_version=([^;]+);", value).group(1)
        assert version_field.startswith(f"{FIXED_VERSION}.")
        assert len(version_field.rsplit(".", 1)[1]) == 3

    def test_derives_from_the_first_user_message_only(self):
        """Stability invariant: appending turns must not change the header.

        This is what keeps the block byte-stable for the life of a
        conversation, so prepending it cannot break prompt caching.
        """
        first_turn = build_billing_header_value(
            [{"role": "user", "content": "the original question"}],
            version=FIXED_VERSION,
        )
        later_turn = build_billing_header_value(
            [
                {"role": "user", "content": "the original question"},
                {"role": "assistant", "content": "an answer"},
                {"role": "user", "content": "a follow-up"},
            ],
            version=FIXED_VERSION,
        )
        assert first_turn == later_turn

    def test_different_first_message_yields_a_different_digest(self):
        one = build_billing_header_value(
            [{"role": "user", "content": "question one"}], version=FIXED_VERSION
        )
        two = build_billing_header_value(
            [{"role": "user", "content": "question two"}], version=FIXED_VERSION
        )
        assert one != two

    def test_reads_text_out_of_block_style_content(self):
        blocks = build_billing_header_value(
            [{"role": "user", "content": [{"type": "text", "text": "hello world"}]}],
            version=FIXED_VERSION,
        )
        plain = build_billing_header_value(
            [{"role": "user", "content": "hello world"}], version=FIXED_VERSION
        )
        assert blocks == plain

    def test_absent_when_the_conversation_has_no_user_turn(self):
        assert (
            build_billing_header_value(
                [{"role": "assistant", "content": "hi"}], version=FIXED_VERSION
            )
            is None
        )

    def test_short_first_message_does_not_raise(self):
        """Sampled char positions run past the end of a short message."""
        value = build_billing_header_value(
            [{"role": "user", "content": "hi"}], version=FIXED_VERSION
        )
        assert value.startswith(BILLING_PREFIX)

    def test_non_ascii_first_message_does_not_raise(self):
        value = build_billing_header_value(
            [{"role": "user", "content": "¿qué tal? 🎉"}], version=FIXED_VERSION
        )
        assert value.startswith(BILLING_PREFIX)


class TestBillingHeaderInOAuthKwargs:
    def test_is_the_first_system_block_ahead_of_the_identity(self):
        """Wire layout must match Claude Code: [billing, identity, ...rest]."""
        kwargs = _oauth_kwargs(
            [
                {"role": "system", "content": "hermes instructions"},
                {"role": "user", "content": "hello"},
            ]
        )
        texts = _system_texts(kwargs)
        assert texts[0].startswith(BILLING_PREFIX)
        assert texts[1] == _CLAUDE_CODE_SYSTEM_PREFIX
        assert "hermes instructions" in texts[2]

    def test_absent_on_api_key_requests(self):
        kwargs = build_anthropic_kwargs(
            model="claude-sonnet-4-6",
            messages=[
                {"role": "system", "content": "hermes instructions"},
                {"role": "user", "content": "hello"},
            ],
            tools=None,
            max_tokens=1024,
            reasoning_config=None,
            is_oauth=False,
        )
        system = kwargs.get("system")
        texts = system if isinstance(system, str) else " ".join(_system_texts(kwargs))
        assert BILLING_PREFIX not in texts

    def test_emitted_when_the_request_has_no_system_prompt(self):
        kwargs = _oauth_kwargs([{"role": "user", "content": "hello"}])
        assert _system_texts(kwargs)[0].startswith(BILLING_PREFIX)

    def test_survives_the_product_name_sanitizer(self):
        """The sanitizer rewrites Hermes branding; it must not touch the header."""
        kwargs = _oauth_kwargs(
            [
                {"role": "system", "content": "You are Hermes Agent by Nous Research"},
                {"role": "user", "content": "hello"},
            ]
        )
        header = _system_texts(kwargs)[0]
        expected = build_billing_header_value(
            [{"role": "user", "content": "hello"}], version=FIXED_VERSION
        )
        assert header == expected

    def test_stable_across_turns_of_one_conversation(self):
        """Prompt-caching invariant at the kwargs level."""
        turn_one = _oauth_kwargs(
            [
                {"role": "system", "content": "hermes instructions"},
                {"role": "user", "content": "the original question"},
            ]
        )
        turn_two = _oauth_kwargs(
            [
                {"role": "system", "content": "hermes instructions"},
                {"role": "user", "content": "the original question"},
                {"role": "assistant", "content": "an answer"},
                {"role": "user", "content": "a follow-up"},
            ]
        )
        assert _system_texts(turn_one)[0] == _system_texts(turn_two)[0]

    def test_tool_name_prefixing_still_applies(self):
        """Guard the sibling OAuth transform the new block sits next to."""
        kwargs = _oauth_kwargs(
            [{"role": "user", "content": "hello"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "read",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        )
        assert kwargs["tools"][0]["name"] == "mcp__read_file"
