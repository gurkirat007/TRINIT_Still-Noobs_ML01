import os
import random
import discord
import chat_bot
import predictor
token = os.getenv("DISCORD_TOKEN")
my_guild = os.getenv("DISCORD_GUILD")

intents = discord.Intents.default()
client = discord.Client(intents=intents)


@client.event
async def on_ready():
    for guild in client.guilds:
        if guild.name == my_guild:
            break

    print(
        f"{client.user} is connected to the following guild:\n"
        f"{guild.name}(id: {guild.id})"
    )

image_types = ["png", "jpeg", "gif", "jpg"]


@client.event
async def on_message(message):
    if message.author == client.user:
        return
    save_folder = ""
    message_content = message.content.lower()
    if message_content:
        reply = chat_bot.predict(message_content)
        await message.channel.send(reply)
    print(len(message.attachments))
    if len(message.attachments) > 0:
        for attachment in message.attachments:
            if any(attachment.filename.lower().endswith(image) for image in image_types):
                save_folder = "user_images/{}".format(attachment.filename)
                await attachment.save(save_folder)

        reply = predictor.predict(save_folder)
        await message.channel.send(reply)
client.run(token)
