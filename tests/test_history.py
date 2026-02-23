"""Tests for Discord history summary helpers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import discord
import pytest

from takopi_discord.history import (
    HistoryMessage,
    build_summary_prompt,
    fetch_recent_history,
    normalize_summary_window,
    summary_window_delta,
)


class _FakeAuthor:
    def __init__(self, name: str, *, bot: bool = False) -> None:
        self.display_name = name
        self.bot = bot


class _FakeAttachment:
    def __init__(self, filename: str) -> None:
        self.filename = filename


class _FakeMessage:
    def __init__(
        self,
        *,
        message_id: int,
        author: _FakeAuthor,
        content: str,
        created_at: datetime,
        message_type: discord.MessageType = discord.MessageType.default,
        attachments: list[_FakeAttachment] | None = None,
    ) -> None:
        self.id = message_id
        self.author = author
        self.clean_content = content
        self.created_at = created_at
        self.type = message_type
        self.attachments = attachments or []
        self.jump_url = f"https://discord.test/messages/{message_id}"


class _FakeChannel:
    def __init__(self, messages: list[_FakeMessage]) -> None:
        self._messages = messages
        self.last_history_kwargs: dict[str, object] | None = None

    def history(
        self,
        *,
        limit: int | None = None,
        before: datetime | None = None,
        after: datetime | None = None,
        around: datetime | None = None,
        oldest_first: bool | None = None,
    ):
        _ = before, around
        self.last_history_kwargs = {
            "limit": limit,
            "after": after,
            "oldest_first": oldest_first,
        }
        ascending = bool(oldest_first)
        ordered = sorted(
            self._messages,
            key=lambda m: m.created_at,
            reverse=not ascending,
        )
        if after is not None:
            ordered = [m for m in ordered if m.created_at > after]
        if limit is not None:
            ordered = ordered[:limit]

        async def _iter():
            for message in ordered:
                yield message

        return _iter()


def test_normalize_window_defaults_to_7d() -> None:
    assert normalize_summary_window(None) == "7d"
    assert normalize_summary_window("bad-value") == "7d"
    assert normalize_summary_window(" 14d ") == "14d"


def test_summary_window_delta() -> None:
    assert summary_window_delta("24h") == timedelta(hours=24)
    assert summary_window_delta("3d") == timedelta(days=3)
    assert summary_window_delta("14d") == timedelta(days=14)


@pytest.mark.anyio
async def test_fetch_recent_history_filters_and_orders() -> None:
    now = datetime(2026, 2, 23, 18, 0, tzinfo=UTC)
    user = _FakeAuthor("alice")
    bot = _FakeAuthor("takopi", bot=True)

    channel = _FakeChannel(
        [
            _FakeMessage(
                message_id=1,
                author=user,
                content="Kickoff decision",
                created_at=now - timedelta(days=6),
            ),
            _FakeMessage(
                message_id=2,
                author=user,
                content="Too old",
                created_at=now - timedelta(days=9),
            ),
            _FakeMessage(
                message_id=3,
                author=bot,
                content="Bot update",
                created_at=now - timedelta(days=2),
            ),
            _FakeMessage(
                message_id=4,
                author=user,
                content="   ",
                created_at=now - timedelta(days=1),
            ),
            _FakeMessage(
                message_id=5,
                author=user,
                content="",
                created_at=now - timedelta(hours=12),
                message_type=discord.MessageType.reply,
                attachments=[_FakeAttachment("plan.md")],
            ),
            _FakeMessage(
                message_id=6,
                author=user,
                content="System event",
                created_at=now - timedelta(hours=6),
                message_type=discord.MessageType.pins_add,
            ),
        ]
    )

    messages = await fetch_recent_history(
        channel,
        window="7d",
        max_messages=50,
        include_bots=False,
        now=now,
    )

    assert [m.message_id for m in messages] == [1, 5]
    assert "attachments: plan.md" in messages[1].content
    assert channel.last_history_kwargs is not None
    assert channel.last_history_kwargs["limit"] == 50
    assert channel.last_history_kwargs["oldest_first"] is True


@pytest.mark.anyio
async def test_fetch_recent_history_can_include_bots() -> None:
    now = datetime(2026, 2, 23, 18, 0, tzinfo=UTC)
    channel = _FakeChannel(
        [
            _FakeMessage(
                message_id=1,
                author=_FakeAuthor("takopi", bot=True),
                content="Bot note",
                created_at=now - timedelta(days=1),
            )
        ]
    )

    messages = await fetch_recent_history(
        channel,
        window="7d",
        include_bots=True,
        now=now,
    )

    assert len(messages) == 1
    assert messages[0].message_id == 1


def test_build_summary_prompt_sections() -> None:
    prompt = build_summary_prompt(
        [
            HistoryMessage(
                message_id=1,
                author_name="alice",
                timestamp=datetime(2026, 2, 22, 10, 0, tzinfo=UTC),
                content="Decided to ship on Friday",
                jump_url="https://discord.test/messages/1",
            )
        ],
        window="7d",
        focus="release status",
    )

    assert "Time window: last 7d." in prompt
    assert "Focus: release status" in prompt
    assert "Key Decisions" in prompt
    assert "Action Items" in prompt
    assert "Transcript:" in prompt
    assert "Decided to ship on Friday" in prompt
