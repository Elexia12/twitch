# bot_local.py
import os
import torch
from twitchio.ext import commands
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Twitch credentials ---
TWITCH_TOKEN = "oauth:your_twitch_oauth_token_here"
TWITCH_NICK = "your_bot_username"
TWITCH_CHANNEL = "your_channel_name"

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
        # Avoid replying to itself
        if message.author.name.lower() == TWITCH_NICK.lower():
            return

        print(f"{message.author.name}: {message.content}")

        # Encode user input
        input_ids = tokenizer.encode(message.content + tokenizer.eos_token, return_tensors="pt")

        # Append chat history (so bot "remembers" a bit)
        global chat_history
        if chat_history:
            bot_input_ids = torch.cat([chat_history, input_ids], dim=-1)
        else:
            bot_input_ids = input_ids

        # Generate a reply
        output_ids = model.generate(
            bot_input_ids,
            max_length=200,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8
        )

        chat_history = output_ids  # save history
        reply = tokenizer.decode(output_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

        # Send reply
        await message.channel.send(f"@{message.author.name} {reply}")

# --- Run bot ---
if __name__ == "__main__":
    bot = Bot()
    bot.run()