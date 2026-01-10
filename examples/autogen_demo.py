#!/usr/bin/env python3
"""
AutoGen Multi-Agent Demo - Two Copilots Collaborating!

This demo shows two AI agents working together:
- Writer: Creates content
- Critic: Reviews and improves it

They discuss until the Critic approves with "APPROVE"
"""
import asyncio
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient


async def main():
    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ Set OPENAI_API_KEY environment variable first!")
        print("   export OPENAI_API_KEY='sk-...'")
        return

    print("🤖🤖 AutoGen Multi-Agent Demo")
    print("=" * 50)
    
    # Create the model client
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",  # Cheap but capable
        api_key=api_key,
    )

    # Agent 1: The Writer
    writer = AssistantAgent(
        name="Writer",
        model_client=model_client,
        system_message="""You are a creative writer. 
        Write concise, engaging content based on the task.
        Keep responses under 100 words.
        Listen to feedback from the Critic and improve your work.""",
    )

    # Agent 2: The Critic  
    critic = AssistantAgent(
        name="Critic",
        model_client=model_client,
        system_message="""You are a constructive critic.
        Review the Writer's work and provide specific feedback.
        If the work is good enough, respond with exactly: APPROVE
        Be concise - max 50 words per review.
        After 2-3 rounds, approve if reasonably good.""",
    )

    # Termination condition: stop when Critic says "APPROVE"
    termination = TextMentionTermination("APPROVE")

    # Create the team - agents take turns
    team = RoundRobinGroupChat(
        participants=[writer, critic],
        termination_condition=termination,
        max_turns=10,  # Safety limit
    )

    # Run the collaboration!
    print("\n📝 Task: Write a haiku about AI agents working together\n")
    print("-" * 50)
    
    result = await Console(
        team.run_stream(task="Write a haiku about AI agents working together")
    )
    
    print("-" * 50)
    print(f"\n✅ Done! Messages exchanged: {len(result.messages)}")


if __name__ == "__main__":
    asyncio.run(main())
