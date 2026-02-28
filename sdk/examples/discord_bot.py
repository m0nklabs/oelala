"""
Oelala Discord Bot Example
==========================

A Discord bot that generates AI images and videos using the Oelala API.

Setup:
    1. pip install discord.py oelala
    2. Create a Discord bot at https://discord.com/developers/applications
    3. Set environment variables:
       - DISCORD_TOKEN: Your Discord bot token
       - OELALA_API_KEY: Your Oelala API key (oelala_...)
    4. python discord_bot.py

Commands:
    /imagine <prompt>         - Generate an image from text
    /video <prompt>           - Generate a video from text
    /animate <image> <prompt> - Animate an uploaded image
    /credits                  - Check your credit balance
"""

import os
import io
import asyncio

import discord
from discord import app_commands

# Install: pip install oelala
from oelala import AsyncOelalaClient

# ── Configuration ────────────────────────────────────────────────

DISCORD_TOKEN = os.environ["DISCORD_TOKEN"]
OELALA_API_KEY = os.environ["OELALA_API_KEY"]

# ── Bot Setup ────────────────────────────────────────────────────

intents = discord.Intents.default()
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)
oelala = AsyncOelalaClient(api_key=OELALA_API_KEY)


@client.event
async def on_ready():
    await tree.sync()
    print(f"✅ Bot ready as {client.user} — {len(client.guilds)} guilds")


# ── /imagine ─────────────────────────────────────────────────────

@tree.command(name="imagine", description="Generate an AI image from a text prompt")
@app_commands.describe(
    prompt="Describe the image you want to create",
    width="Image width (256-2048, default 1024)",
    height="Image height (256-2048, default 1024)",
)
async def imagine(
    interaction: discord.Interaction,
    prompt: str,
    width: int = 1024,
    height: int = 1024,
):
    await interaction.response.defer(thinking=True)

    try:
        # Submit job
        job = await oelala.text_to_image(prompt, width=width, height=height)
        await interaction.followup.send(
            f"🎨 Generating image...\n> {prompt}\n"
            f"Job `{job.job_id}` — estimated ~{job.estimated_time_seconds or '?'}s"
        )

        # Wait for completion
        result = await oelala.wait_for_job(job.job_id, poll_interval=3, timeout=300)

        if result.succeeded and result.result_url:
            embed = discord.Embed(
                title="🖼️ Image Generated",
                description=f"**Prompt:** {prompt}",
                color=0x7C3AED,
            )
            embed.set_image(url=result.result_url)
            embed.set_footer(text=f"Job: {job.job_id}")
            await interaction.followup.send(embed=embed)
        else:
            await interaction.followup.send(
                f"❌ Generation failed: {result.error or 'Unknown error'}"
            )

    except Exception as e:
        await interaction.followup.send(f"❌ Error: {e}")


# ── /video ───────────────────────────────────────────────────────

@tree.command(name="video", description="Generate an AI video from a text prompt")
@app_commands.describe(
    prompt="Describe the video you want to create",
    duration="Video duration in seconds (1-30, default 5)",
)
async def video(
    interaction: discord.Interaction,
    prompt: str,
    duration: int = 5,
):
    await interaction.response.defer(thinking=True)

    try:
        job = await oelala.text_to_video(
            prompt,
            duration_seconds=duration,
            width=848,
            height=480,
        )
        await interaction.followup.send(
            f"🎬 Generating video ({duration}s)...\n> {prompt}\n"
            f"Job `{job.job_id}` — estimated ~{job.estimated_time_seconds or '?'}s\n"
            f"⏳ This may take a few minutes."
        )

        # Poll with progress updates
        last_progress = -1

        async def on_progress(status):
            nonlocal last_progress
            if status.progress and status.progress != last_progress:
                last_progress = status.progress

        result = await oelala.wait_for_job(
            job.job_id, poll_interval=5, timeout=600, on_progress=on_progress
        )

        if result.succeeded and result.result_url:
            embed = discord.Embed(
                title="🎬 Video Generated",
                description=f"**Prompt:** {prompt}\n**Duration:** {duration}s",
                color=0x7C3AED,
            )
            embed.add_field(name="Download", value=f"[Click here]({result.result_url})")
            embed.set_footer(text=f"Job: {job.job_id}")
            await interaction.followup.send(embed=embed)
        else:
            await interaction.followup.send(
                f"❌ Video generation failed: {result.error or 'Unknown error'}"
            )

    except Exception as e:
        await interaction.followup.send(f"❌ Error: {e}")


# ── /animate ─────────────────────────────────────────────────────

@tree.command(name="animate", description="Animate an uploaded image into a video")
@app_commands.describe(
    image="Upload the image to animate",
    prompt="Describe how the image should be animated",
    duration="Video duration in seconds (1-30, default 5)",
)
async def animate(
    interaction: discord.Interaction,
    image: discord.Attachment,
    prompt: str,
    duration: int = 5,
):
    await interaction.response.defer(thinking=True)

    if not image.content_type or not image.content_type.startswith("image/"):
        await interaction.followup.send("❌ Please upload an image file.")
        return

    try:
        job = await oelala.image_to_video(
            prompt,
            image_url=image.url,
            duration_seconds=duration,
            width=848,
            height=480,
        )
        await interaction.followup.send(
            f"🎞️ Animating image ({duration}s)...\n> {prompt}\n"
            f"Job `{job.job_id}` — estimated ~{job.estimated_time_seconds or '?'}s"
        )

        result = await oelala.wait_for_job(job.job_id, poll_interval=5, timeout=600)

        if result.succeeded and result.result_url:
            embed = discord.Embed(
                title="🎞️ Animation Complete",
                description=f"**Prompt:** {prompt}",
                color=0x7C3AED,
            )
            embed.set_thumbnail(url=image.url)
            embed.add_field(name="Download", value=f"[Click here]({result.result_url})")
            embed.set_footer(text=f"Job: {job.job_id}")
            await interaction.followup.send(embed=embed)
        else:
            await interaction.followup.send(
                f"❌ Animation failed: {result.error or 'Unknown error'}"
            )

    except Exception as e:
        await interaction.followup.send(f"❌ Error: {e}")


# ── /credits ─────────────────────────────────────────────────────

@tree.command(name="credits", description="Check your Oelala credit balance")
async def credits(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)

    try:
        balance = await oelala.get_credits()
        embed = discord.Embed(title="💰 Credit Balance", color=0x7C3AED)
        embed.add_field(name="Available", value=f"{balance.balance:,}", inline=True)
        embed.add_field(name="Total Used", value=f"{balance.lifetime_used:,}", inline=True)
        embed.add_field(name="Total Purchased", value=f"{balance.lifetime_purchased:,}", inline=True)
        await interaction.followup.send(embed=embed, ephemeral=True)
    except Exception as e:
        await interaction.followup.send(f"❌ Error: {e}", ephemeral=True)


# ── Run ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    client.run(DISCORD_TOKEN)
