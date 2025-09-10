from twitchio.ext import commands
import logging

logging.basicConfig(level=logging.DEBUG)

# -------------- CONFIG ----------------
TWITCH_TOKEN = "oauth:80c4uc4a4pcp5m4fg96rnlk23rybv9"
CHANNEL = "elexiacr"  # lowercase
PREFIX = "!"
# --------------------------------------

class Bot(commands.Bot):

    def __init__(self):
        super().__init__(
            token=TWITCH_TOKEN,
            prefix=PREFIX,
            initial_channels=[CHANNEL]
        )

    async def event_ready(self):
        print(f"✅ Logged in as {self.nick}")
        print(f"✅ Joined channels: {self.connected_channels}")

    async def event_message(self, message):
        # ignore messages from the bot itself
        if message.echo:
            return

        # print all messages in console
        print(f"[{message.channel.name}] {message.author.name}: {message.content}")

        # handle commands
        await self.handle_commands(message)

    @commands.command()
    async def ping(self, ctx):
        await ctx.send(f"Pong {ctx.author.name}!")

if __name__ == "__main__":
    bot = Bot()
    bot.run()