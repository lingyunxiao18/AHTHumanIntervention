import os
import json
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
        "You are observing your confederate agent's behavior, which has recently changed unexpectedly. "
        "A human controller is providing high-level intervention instructions to adjust your strategy. "
        "Based on the following human command, decide whether you need to change your heuristic. "
        "If a change is required, output a JSON object with one key, 'ego_agent_new_heuristic', "
        "with its value set to either 'clockwise' or 'counterclockwise'. If no change is necessary, output null for the value. "
        "Do not include any additional commentary or keys; output valid JSON only.\n\n"
        f"Human command: \"{user_command}\""
    )

    try:
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
