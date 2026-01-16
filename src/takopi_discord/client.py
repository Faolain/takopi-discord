"""Discord API client wrapper."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import discord

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    MessageHandler = Callable[[discord.Message], Coroutine[Any, Any, None]]
    InteractionHandler = Callable[[discord.Interaction], Coroutine[Any, Any, None]]


@dataclass(frozen=True, slots=True)
class SentMessage:
    """Result of sending a message."""

    message_id: int
    channel_id: int
    thread_id: int | None = None


class DiscordBotClient:
    """Wrapper around Pycord Bot for takopi integration."""

    def __init__(self, token: str, *, guild_id: int | None = None) -> None:
        self._token = token
        self._guild_id = guild_id
        self._message_handler: MessageHandler | None = None
        self._interaction_handler: InteractionHandler | None = None
        # Defer bot creation until inside async context (Python 3.10+ compatibility)
        self._bot: discord.Bot | None = None
        self._ready_event: asyncio.Event | None = None

    def _ensure_bot(self) -> discord.Bot:
        """Create the bot if not already created. Must be called from async context."""
        if self._bot is not None:
            return self._bot

        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.members = False
        # Required for receiving messages in threads
        intents.messages = True
        # Use discord.Bot which has built-in slash command support
        # debug_guilds ensures instant command sync to specific guilds
        debug_guilds = [self._guild_id] if self._guild_id else None
        self._bot = discord.Bot(intents=intents, debug_guilds=debug_guilds)
        self._ready_event = asyncio.Event()

        @self._bot.event
        async def on_ready() -> None:
            assert self._ready_event is not None
            self._ready_event.set()

        @self._bot.event
        async def on_message(message: discord.Message) -> None:
            assert self._bot is not None
            # Debug: print to stdout to bypass any logging issues
            print(
                f"[DEBUG on_message] channel={type(message.channel).__name__} id={message.channel.id} author={message.author.name} content={message.content[:30] if message.content else '(empty)'}",
                flush=True,
            )
            # Debug: log ALL incoming messages at the client level
            import logging

            logging.getLogger("takopi.discord.client").debug(
                "on_message raw: channel_type=%s channel_id=%s author=%s content_preview=%s",
                type(message.channel).__name__,
                message.channel.id,
                message.author.name,
                message.content[:30] if message.content else "(empty)",
            )
            if message.author == self._bot.user:
                print("[DEBUG on_message] SKIPPED: bot's own message", flush=True)
                return
            if self._message_handler is not None:
                print("[DEBUG on_message] calling message_handler...", flush=True)
                await self._message_handler(message)
                print("[DEBUG on_message] message_handler returned", flush=True)
            else:
                print("[DEBUG on_message] NO message_handler set!", flush=True)

        @self._bot.event
        async def on_application_command_error(
            ctx: discord.ApplicationContext, error: Exception
        ) -> None:
            print(f"[DEBUG] Command error in /{ctx.command.name}: {type(error).__name__}: {error}", flush=True)
            import traceback
            traceback.print_exc()

        @self._bot.event
        async def on_application_command(ctx: discord.ApplicationContext) -> None:
            print(f"[DEBUG] Command invoked: /{ctx.command.name}", flush=True)

        @self._bot.event
        async def on_interaction(interaction: discord.Interaction) -> None:
            print(f"[DEBUG] Interaction received: type={interaction.type} data={interaction.data}", flush=True)

        return self._bot

    @property
    def bot(self) -> discord.Bot:
        """Get the underlying Pycord bot. Creates it if needed."""
        return self._ensure_bot()

    @property
    def user(self) -> discord.User | None:
        """Get the bot user."""
        if self._bot is None:
            return None
        return self._bot.user

    def set_message_handler(self, handler: MessageHandler) -> None:
        """Set the message handler."""
        self._message_handler = handler

    def set_interaction_handler(self, handler: InteractionHandler) -> None:
        """Set the interaction handler for non-command interactions."""
        self._interaction_handler = handler

    async def start(self) -> None:
        """Start the bot and wait until ready."""
        bot = self._ensure_bot()
        assert self._ready_event is not None

        # Debug: print pending commands before starting
        print(f"[DEBUG] Pending commands before start: {len(bot.pending_application_commands)}", flush=True)
        for cmd in bot.pending_application_commands:
            print(f"[DEBUG]   - {cmd.name}", flush=True)

        asyncio.create_task(bot.start(self._token))
        await self._ready_event.wait()

        # Debug: print pending commands after ready
        print(f"[DEBUG] Pending commands after ready: {len(bot.pending_application_commands)}", flush=True)
        for cmd in bot.pending_application_commands:
            print(f"[DEBUG]   - {cmd.name}: {cmd.options}", flush=True)

        # Pycord auto-syncs commands on_connect, so we don't need to manually sync.
        # Manual sync was causing hangs. If guild-specific fast sync is needed,
        # set debug_guilds on the Bot instead.
        print("[DEBUG] Bot ready, commands should be auto-synced by Pycord", flush=True)

    async def close(self) -> None:
        """Close the bot connection."""
        if self._bot is not None:
            await self._bot.close()

    async def wait_until_ready(self) -> None:
        """Wait until the bot is ready."""
        self._ensure_bot()
        assert self._ready_event is not None
        await self._ready_event.wait()

    async def send_message(
        self,
        *,
        channel_id: int,
        content: str,
        reply_to_message_id: int | None = None,
        thread_id: int | None = None,
        view: discord.ui.View | None = None,
        embed: discord.Embed | None = None,
    ) -> SentMessage | None:
        """Send a message to a channel."""
        channel = self._bot.get_channel(thread_id or channel_id)
        if channel is None:
            try:
                channel = await self._bot.fetch_channel(thread_id or channel_id)
            except discord.NotFound:
                return None

        if not isinstance(channel, discord.abc.Messageable):
            return None

        reference = None
        if reply_to_message_id is not None:
            # Use thread_id for the reference channel if we're in a thread,
            # since that's where the original message actually exists
            reference = discord.MessageReference(
                message_id=reply_to_message_id,
                channel_id=thread_id or channel_id,
            )

        try:
            kwargs: dict[str, Any] = {"content": content}
            if reference is not None:
                kwargs["reference"] = reference
            if view is not None:
                kwargs["view"] = view
            if embed is not None:
                kwargs["embed"] = embed

            message = await channel.send(**kwargs)
            return SentMessage(
                message_id=message.id,
                channel_id=message.channel.id,
                thread_id=thread_id,
            )
        except discord.HTTPException:
            # If send failed and we had a reference, retry without it
            # This handles cases like new threads where the reply message
            # might not be in the thread
            if reference is not None:
                try:
                    kwargs.pop("reference", None)
                    message = await channel.send(**kwargs)
                    return SentMessage(
                        message_id=message.id,
                        channel_id=message.channel.id,
                        thread_id=thread_id,
                    )
                except discord.HTTPException:
                    return None
            return None

    async def edit_message(
        self,
        *,
        channel_id: int,
        message_id: int,
        content: str,
        view: discord.ui.View | None = None,
        embed: discord.Embed | None = None,
    ) -> SentMessage | None:
        """Edit an existing message."""
        channel = self._bot.get_channel(channel_id)
        if channel is None:
            try:
                channel = await self._bot.fetch_channel(channel_id)
            except discord.NotFound:
                return None

        if not isinstance(channel, discord.abc.Messageable):
            return None

        try:
            message = await channel.fetch_message(message_id)
            kwargs: dict[str, Any] = {"content": content}
            if view is not None:
                kwargs["view"] = view
            if embed is not None:
                kwargs["embed"] = embed

            edited = await message.edit(**kwargs)
            return SentMessage(
                message_id=edited.id,
                channel_id=edited.channel.id,
            )
        except discord.HTTPException:
            return None

    async def delete_message(
        self,
        *,
        channel_id: int,
        message_id: int,
    ) -> bool:
        """Delete a message."""
        channel = self._bot.get_channel(channel_id)
        if channel is None:
            try:
                channel = await self._bot.fetch_channel(channel_id)
            except discord.NotFound:
                return False

        if not isinstance(channel, discord.abc.Messageable):
            return False

        try:
            message = await channel.fetch_message(message_id)
            await message.delete()
            return True
        except discord.HTTPException:
            return False

    async def create_thread(
        self,
        *,
        channel_id: int,
        message_id: int,
        name: str,
        auto_archive_duration: int = 1440,  # 24 hours
    ) -> int | None:
        """Create a thread from a message."""
        channel = self._bot.get_channel(channel_id)
        if channel is None:
            try:
                channel = await self._bot.fetch_channel(channel_id)
            except discord.NotFound:
                return None

        if not isinstance(channel, discord.TextChannel):
            return None

        try:
            message = await channel.fetch_message(message_id)
            thread = await message.create_thread(
                name=name,
                auto_archive_duration=auto_archive_duration,
            )
            # Join the thread so we receive messages from it
            await thread.join()
            return thread.id
        except discord.HTTPException:
            return None

    def get_guild(self, guild_id: int) -> discord.Guild | None:
        """Get a guild by ID."""
        return self._bot.get_guild(guild_id)

    def get_channel(self, channel_id: int) -> discord.abc.GuildChannel | None:
        """Get a channel by ID."""
        channel = self._bot.get_channel(channel_id)
        if isinstance(channel, discord.abc.GuildChannel):
            return channel
        return None
