import sys, os
import time
import logging
from twitchio.ext import commands
from openai import OpenAI, APIError, RateLimitError, APITimeoutError
from dotenv import load_dotenv
from datetime import datetime
from collections import defaultdict, deque

logging.basicConfig(level=logging.DEBUG)
load_dotenv()

# ---------------- CONFIG ----------------
TWITCH_TOKEN = os.getenv("TWITCH_TOKEN")
CHANNEL = os.getenv("TWITCH_CHANNEL")  # lowercase
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PREFIX = "!"
MAX_MEMORY = 5  # how many past exchanges to keep per channel
# ----------------------------------------

# Init OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# Conversation history per channel (short-term memory)
memory = defaultdict(lambda: deque(maxlen=MAX_MEMORY))

# Load your content (long-term "knowledge")
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "my_content.txt")

def resource_path(relative_path):
    """Prefer the .exe folder, fallback to PyInstaller temp folder."""
    if hasattr(sys, "_MEIPASS"):
        # When running as .exe, prefer the folder where the exe is located
        base_path = os.path.dirname(sys.executable)
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

try:
    with open(resource_path("my_content.txt"), "r", encoding="utf-8") as f:
        custom_knowledge = f.read()
except FileNotFoundError:
    print(f"⚠️ Could not find my_content.txt at: {file_path}")
    custom_knowledge = ""

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
        if message.echo:
            return

        print(f"[{message.channel.name}] {message.author.name}: {message.content}")
        await self.handle_commands(message)

    @commands.command()
    async def ping(self, ctx):
        await ctx.send(f"Pong {ctx.author.name}!")

    @commands.command()
    async def ask(self, ctx):
        """Ask the AI a question"""
        prompt = ctx.message.content[len("!ask "):].strip()
        if not prompt:
            await ctx.send("⚠️ Usage: !ask <your question>")
            return

        await ctx.send("EleGPT esta pensando...")

        today = datetime.now().strftime("%d %B %Y")

        # System message includes real date + channel context
        system_message = (
            f"You are a helpful assistant answering in English or Spanish. "
            f"Today's real date is {today}. Always use this date when asked about 'today'. "
            f"You are currently chatting in the Twitch channel '{ctx.channel.name}', "
            f"and the streamer is '{CHANNEL}'. "
            f"Keep context from prior exchanges if relevant."
        )

        # Add custom knowledge as if it’s a special "assistant note"
        knowledge_message = {
            "role": "system",
            "content": (
                f"Here is additional long-term knowledge about the streamer and channel: "
                f"{custom_knowledge}"
            )
        }

        # Build conversation with memory
        conversation = [{"role": "system", "content": system_message}]
        conversation.append(knowledge_message)
        conversation.extend(memory[ctx.channel.name])
        conversation.append({"role": "user", "content": prompt})

        try:
            start_time = time.time()

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=conversation,
                timeout=30
            )

            elapsed = time.time() - start_time
            logging.info(f"✅ OpenAI responded in {elapsed:.2f}s")

            answer = response.choices[0].message.content.strip()

            # Update memory
            memory[ctx.channel.name].append({"role": "user", "content": prompt})
            memory[ctx.channel.name].append({"role": "assistant", "content": answer})

            twitch_limit = 500

            # Split the answer in bits so that the full message of the bot is shown
            if (len(answer) > twitch_limit):
                await ctx.send(answer[:twitch_limit])
                await ctx.send(answer[twitch_limit:])
            else:
                await ctx.send(answer)  # Twitch limit

        except APITimeoutError:
            elapsed = time.time() - start_time
            logging.error(f"❌ OpenAI timed out after {elapsed:.2f}s")
            await ctx.send("⚠️ Error: OpenAI API timed out.")

        except APIError as e:
            elapsed = time.time() - start_time
            logging.error(f"❌ OpenAI API error after {elapsed:.2f}s: {e}")
            await ctx.send("⚠️ Error: OpenAI API failed.")

        except Exception as e:
            logging.error(f"❌ Twitch or network error: {e}")
            await ctx.send("⚠️ Error: Twitch or network issue.")


if __name__ == "__main__":
    bot = Bot()
    bot.run()
