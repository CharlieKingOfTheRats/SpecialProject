import openai
from openai import tool
import asyncio
import tiktoken  # For token counting
import os

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define tools
@tool
def get_wine_pairing(food: str) -> str:
    """Suggest a wine pairing based on the given food."""
    if "steak" in food.lower():
        return "A bold Cabernet Sauvignon pairs well with steak."
    elif "salmon" in food.lower():
        return "Try a Pinot Noir or Chardonnay with salmon."
    elif "spicy" in food.lower():
        return "A slightly sweet Riesling or Gewürztraminer complements spicy dishes."
    else:
        return f"A Sauvignon Blanc is a safe choice with {food}."

@tool
def calculate_percentage(base: float, percent: float) -> float:
    """Calculate a percentage of a number."""
    return base * (percent / 100)

# Token counting utility
def count_tokens(text, model="gpt-4"):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

# Reason about the prompt and generate refined system instructions
def reflect_and_refine(user_prompt: str) -> str:
    if any(word in user_prompt.lower() for word in ["wine", "pairing", "food"]):
        return (
            "You are a culinary assistant that specializes in wine and food pairings. "
            "You understand food flavor profiles, wine regions, and tasting notes. "
            f"Your goal is to help the user based on their query: '{user_prompt}'"
        )
    elif "percent" in user_prompt or "calculate" in user_prompt:
        return (
            "You are a helpful assistant focused on numerical and financial reasoning. "
            "Use your tools to calculate percentages or breakdowns accurately. "
            f"Start by understanding the query: '{user_prompt}'"
        )
    else:
        return (
            "You are a general-purpose intelligent assistant. "
            "Use available tools to provide insightful, complete, and helpful answers. "
            f"Consider the user's query: '{user_prompt}'"
        )

# Create the agent
def create_agent(user_prompt: str):
    system_prompt = reflect_and_refine(user_prompt)
    print(f"\n[System Prompt Created]:\n{system_prompt}\n")
    return openai.Agent.create(
        name="ReflectiveDynamicAgent",
        instructions=system_prompt,
        tools=[get_wine_pairing, calculate_percentage],
        model="gpt-4-turbo"
    ), system_prompt

# Run the agent
async def run_agent(agent, user_prompt, system_prompt):
    thread = openai.Thread.create()
    
    # Send message
    user_message = {"role": "user", "content": user_prompt}
    run = openai.Agent.run(
        agent_id=agent.id,
        thread_id=thread.id,
        instructions=system_prompt,
        messages=[user_message],
        stream=False,
    )

    # Wait until run is complete
    completed_run = openai.AgentRun.retrieve(run.id, thread_id=thread.id)
    
    # Get all messages
    messages = openai.Thread.messages(thread_id=thread.id)
    for msg in messages:
        if msg.role == "assistant":
            response = msg.content[0]["text"]["value"]
            print(f"\n[Agent Response]:\n{response}\n")
            return response, user_message["content"]

# Entry point
if __name__ == "__main__":
    user_input = input("Enter a prompt: ").strip()
    agent, system_prompt = create_agent(user_input)

    response, original_prompt = asyncio.run(run_agent(agent, user_input, system_prompt))

    total_tokens = count_tokens(system_prompt + original_prompt + response)
    print(f"[Token Usage Estimate]: {total_tokens} tokens (approx)")