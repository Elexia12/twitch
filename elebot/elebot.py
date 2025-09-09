# bot_local.py
import os
import torch
from twitchio.ext import commands
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()
TWITCH_TOKEN = os.getenv("TWITCH_TOKEN")
TWITCH_NICK = os.getenv("TWITCH_NICK")
TWITCH_CHANNEL = os.getenv("TWITCH_CHANNEL")

# --- Load DialoGPT model ---
print("Loading DialoGPT model...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Keep track of conversation history
chat_history = []

class Bot(commands.Bot):

    def __init__(self):
        super().__init__(token=TWITCH_TOKEN, prefix="!", initial_channels=[TWITCH_CHANNEL])

    async def event_ready(self):
        print(f"Bot {TWITCH_NICK} is online in {TWITCH_CHANNEL}'s chat!")

    async def event_message(self, message):
        if message.author.name.lower() == TWITCH_NICK.lower():
            return  # ignore itself

        content = message.content.strip()

        # Check if it's an @mention or a command
        mentioned = f"@{TWITCH_NICK.lower()}" in content.lower()
        is_command = content.lower().startswith("!askbot")

        if not (mentioned or is_command):
            return  # ignore everything else

        print(f"{message.author.name} triggered the bot: {content}")

        # Clean up input (remove mention or command)
        clean_text = content
        clean_text = clean_text.replace(f"@{TWITCH_NICK}", "").strip()
        if is_command:
            clean_text = clean_text[len("!askbot"):].strip()

        # Encode user input
        input_ids = tokenizer.encode(clean_text + tokenizer.eos_token, return_tensors="pt")

        global chat_history
        if chat_history:
            bot_input_ids = torch.cat([chat_history, input_ids], dim=-1)
        else:
            bot_input_ids = input_ids

        # Generate reply
        output_ids = model.generate(
            bot_input_ids,
            max_length=200,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8
        )

        chat_history = output_ids
        reply = tokenizer.decode(output_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

        # Reply to user
        await message.channel.send(f"@{message.author.name} {reply}")

if __name__ == "__main__":
    bot = Bot()
    bot.run()