"""Tests for default values and fallbacks."""

from __future__ import annotations

import json

import pytest

from takopi_discord.state import DiscordStateStore
from takopi_discord.types import DiscordChannelContext


def test_channel_context_default_base_branch_is_main() -> None:
    ctx = DiscordChannelContext(project="~/dev/example")
    assert ctx.worktree_base == "main"


@pytest.mark.anyio
async def test_state_store_defaults_worktree_base_to_main_when_missing(
    tmp_path,
) -> None:
    config_path = tmp_path / "takopi.toml"
    state_path = tmp_path / "discord_state.json"

    state_path.write_text(
        json.dumps(
            {
                "version": 2,
                "channels": {
                    "123:456": {
                        "context": {
                            "project": "~/dev/example",
                        }
                    }
                },
                "guilds": {},
            }
        ),
        encoding="utf-8",
    )

    store = DiscordStateStore(config_path=config_path)
    context = await store.get_context(123, 456)

    assert context is not None
    assert isinstance(context, DiscordChannelContext)
    assert context.worktree_base == "main"
