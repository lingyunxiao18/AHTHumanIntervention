import os
import json
from openai import OpenAI

# Initialize OpenAI client only if API key is available
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    client = OpenAI(api_key=api_key)
else:
    client = None
    print("Warning: OPENAI_API_KEY not set. LLM interventions will not work.")

# Ensure your OpenAI API key is set in the environment variable 'OPENAI_API_KEY'

def process_command(user_command: str) -> dict:
    """
    Uses OpenAI's GPT-4.1 nano to process a human intervention command.
    Based on the input command, outputs a JSON object with a single key "ego_agent_new_heuristic".
    The value is expected to be either "clockwise", "counterclockwise", or null if no change is needed.
    
    Example output:
       {"ego_agent_new_heuristic": "counterclockwise"}
    """
    prompt = (
        "You are the ego agent in an Overcooked simulation. "
        "You are collaborating with another agent (the confederate) to efficiently complete cooking tasks. "
        "You use a heuristic-based movement strategy (either clockwise or counterclockwise) to navigate the kitchen. "
        "A human controller provides high-level commands to help you adjust your strategy if needed. "
        "Carefully read the human command provided below. "
        "Decide if this command requires you to change your current heuristic strategy to better align with your teammate's actions. "
        "If a change is necessary, output a JSON object with a single key, 'ego_agent_new_heuristic', "
        "and set its value to either 'clockwise' or 'counterclockwise'. "
        "If no change is necessary based on the command, output the value null. "
        "Output valid JSON only, with no additional commentary or keys.\n\n"
        f"Human command: \"{user_command}\""
    )


    try:
        if client is None:
            print("OpenAI client not initialized. Returning null heuristic.")
            return {"ego_agent_new_heuristic": None}
            
        response = client.chat.completions.create(
            model="gpt-4.1-nano",  # Updated to use GPT-4.1 nano model.
            messages=[
                {"role": "system", "content": "You are a strategic assistant working as the ego agent in an Overcooked simulation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        content = response.choices[0].message.content
        result = json.loads(content)
    except Exception as e:
        print("Error during LLM processing:", e)
        result = {"ego_agent_new_heuristic": None}
    return result
