"""Helpers for fetching and preparing Discord message history."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol

import discord

SUMMARY_WINDOW_CHOICES: tuple[str, ...] = ("24h", "3d", "7d", "14d")
_WINDOW_TO_DELTA: dict[str, timedelta] = {
    "24h": timedelta(hours=24),
    "3d": timedelta(days=3),
    "7d": timedelta(days=7),
    "14d": timedelta(days=14),
}

_SUMMARIZABLE_MESSAGE_TYPES = {
    discord.MessageType.default,
    discord.MessageType.reply,
}


@dataclass(frozen=True, slots=True)
class HistoryMessage:
    """Normalized message format for summary prompts."""

    message_id: int
    author_name: str
    timestamp: datetime
    content: str
    jump_url: str | None


class _HistoryChannel(Protocol):
    """Protocol for channels/threads that support history iteration."""

    def history(
        self,
        *,
        limit: int | None = ...,
        before: datetime | None = ...,
        after: datetime | None = ...,
        around: datetime | None = ...,
        oldest_first: bool | None = ...,
    ): ...


def normalize_summary_window(window: str | None) -> str:
    """Normalize the summary window option, defaulting to 7d."""
    if window is None:
        return "7d"
    normalized = window.strip().lower()
    if normalized in _WINDOW_TO_DELTA:
        return normalized
    return "7d"


def summary_window_delta(window: str | None) -> timedelta:
    """Return timedelta for a summary window string."""
    return _WINDOW_TO_DELTA[normalize_summary_window(window)]


def _render_message_content(
    message: discord.Message,
    *,
    max_chars: int = 600,
) -> str | None:
    text = message.clean_content.strip()
    parts: list[str] = []
    if text:
        parts.append(text.replace("\n", " ").strip())

    if message.attachments:
        names = [a.filename for a in message.attachments if getattr(a, "filename", "")]
        if names:
            parts.append(f"[attachments: {', '.join(names)}]")
        else:
            parts.append(f"[attachments: {len(message.attachments)} file(s)]")

    if not parts:
        return None

    rendered = " ".join(parts).strip()
    if len(rendered) <= max_chars:
        return rendered
    return rendered[: max_chars - 1] + "…"


async def fetch_recent_history(
    channel: _HistoryChannel,
    *,
    window: str = "7d",
    max_messages: int = 200,
    include_bots: bool = False,
    now: datetime | None = None,
    skip_message_ids: set[int] | None = None,
) -> list[HistoryMessage]:
    """Fetch and normalize recent messages from a channel or thread."""
    cutoff = (now or datetime.now(UTC)) - summary_window_delta(window)
    skip_ids = skip_message_ids or set()
    messages: list[HistoryMessage] = []

    history_iter = channel.history(
        limit=max_messages,
        after=cutoff,
        oldest_first=True,
    )
    async for message in history_iter:
        if message.id in skip_ids:
            continue
        if message.type not in _SUMMARIZABLE_MESSAGE_TYPES:
            continue
        if message.author.bot and not include_bots:
            continue

        rendered = _render_message_content(message)
        if rendered is None:
            continue

        created_at = message.created_at
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)

        messages.append(
            HistoryMessage(
                message_id=message.id,
                author_name=message.author.display_name,
                timestamp=created_at,
                content=rendered,
                jump_url=getattr(message, "jump_url", None),
            )
        )

    return messages


def build_summary_prompt(
    messages: list[HistoryMessage],
    *,
    window: str = "7d",
    focus: str | None = None,
) -> str:
    """Build a structured summarization prompt from normalized messages."""
    normalized_window = normalize_summary_window(window)
    focus_text = (focus or "").strip()

    transcript_lines: list[str] = []
    for message in messages:
        ts = message.timestamp.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")
        line = f"[{ts}] {message.author_name}: {message.content}"
        if message.jump_url:
            line = f"{line} ({message.jump_url})"
        transcript_lines.append(line)

    focus_line = (
        f"Focus: {focus_text}"
        if focus_text
        else "Focus: general project/channel catch-up."
    )

    return (
        "You are summarizing a Discord conversation.\n"
        f"Time window: last {normalized_window}.\n"
        f"Messages provided: {len(messages)}.\n"
        f"{focus_line}\n\n"
        "Return a concise summary with these sections:\n"
        "1. Key Decisions\n"
        "2. Action Items (owner + due date if present)\n"
        "3. Blockers/Risks\n"
        "4. Open Questions\n"
        "5. Notable Links/Files\n\n"
        "If a section has no content, write 'None'.\n\n"
        "Transcript:\n"
        + "\n".join(transcript_lines)
    )
