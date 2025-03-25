import openai
from openai import AssistantEventHandler, tool
import asyncio

# Set your OpenAI API key
openai.api_key = "sk-..."

# Step 1: Define some example tools
@tool
def get_wine_pairing(food: str) -> str:
    """Suggest a wine pairing based on the given food."""
    if "steak" in food.lower():
        return "A bold Cabernet Sauvignon pairs well with steak."
    elif "salmon" in food.lower():
        return "Try a Pinot Noir or Chardonnay with salmon."
    else:
        return f"I'm not sure what wine goes with {food}, but a Sauvignon Blanc is usually safe."

@tool
def calculate_percentage(base: float, percent: float) -> float:
    """Calculate a percentage of a number."""
    return base * (percent / 100)

# Step 2: Generate a custom system prompt dynamically
def generate_system_prompt(user_prompt: str) -> str:
    return f"You are a helpful assistant that responds to: '{user_prompt}' with relevant tools and explanations."

# Step 3: Create the agent
def create_agent_from_prompt(user_prompt: str):
    system_prompt = generate_system_prompt(user_prompt)

    return openai.Agent.create(
        name="DynamicCustomAgent",
        instructions=system_prompt,
        tools=[get_wine_pairing, calculate_percentage],
        model="gpt-4-turbo"
    )

# Step 4: Run the agent
async def run_agent(agent, user_prompt):
    thread = openai.Thread.create()
    run = openai.Agent.run(
        agent_id=agent.id,
        thread_id=thread.id,
        instructions=agent.instructions,
        messages=[{"role": "user", "content": user_prompt}],
        stream=False,
    )

    # Get and return the assistant's response
    completed_run = openai.AgentRun.retrieve(run.id, thread_id=thread.id)
    messages = openai.Thread.messages(thread_id=thread.id)
    for msg in messages:
        if msg.role == "assistant":
            print(f"Assistant: {msg.content[0]['text']['value']}")

# Step 5: Main entry
if __name__ == "__main__":
    user_input = "What wine should I serve with spicy Thai noodles?"
    agent = create_agent_from_prompt(user_input)
    asyncio.run(run_agent(agent, user_input))