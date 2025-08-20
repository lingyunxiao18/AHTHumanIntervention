import os
import json
from openai import OpenAI

# Initialize OpenAI client only if API key is available
api_key = os.getenv("OPENAI_API_KEY")

# Fallback: If you want to hardcode the API key for testing (not recommended for production)
if not api_key:
    # Uncomment and set your API key here for testing
    api_key = "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"
    pass

if api_key:
    client = OpenAI(api_key=api_key)
else:
    client = None
    print("Warning: OPENAI_API_KEY not set. LLM interventions will not work.")
    print("To fix this, either:")
    print("1. Set the environment variable: export OPENAI_API_KEY='your-api-key'")
    print("2. Or uncomment and set the api_key variable in this file")

def process_command(user_command: str) -> dict:
    """
    Uses OpenAI's GPT-4.1 nano to process a human intervention command.
    Based on the input command, outputs a JSON object with a single key "ego_agent_new_heuristic".
    The value can be one of the following heuristic strategies:
    
    Movement-based strategies:
    - "clockwise": Move in a clockwise pattern around the kitchen while interacting with objects
    - "counterclockwise": Move in a counterclockwise pattern around the kitchen while interacting with objects
    
    Task-specific strategies:
    - "onion_to_pot": Focus on picking up onions and placing them in pots
    - "plate_agent": Focus on picking up plates and delivering soups
    - null: No change needed in current strategy
    
    Example outputs:
       {"ego_agent_new_heuristic": "clockwise"}
       {"ego_agent_new_heuristic": "onion_to_pot"}
       {"ego_agent_new_heuristic": null}
    """
    prompt = (
        "You are the ego agent in an Overcooked simulation. "
        "You are collaborating with another agent (the confederate) to efficiently complete cooking tasks. "
        "You can use various heuristic-based strategies:\n\n"
        "Movement-based strategies:\n"
        "- clockwise: Move in a clockwise pattern around the kitchen while interacting with objects\n"
        "- counterclockwise: Move in a counterclockwise pattern around the kitchen while interacting with objects\n\n"
        "Task-specific strategies:\n"
        "- onion_to_pot: Focus on picking up onions and placing them in pots\n"
        "- plate_agent: Focus on picking up plates and delivering soups\n\n"
        "A human controller provides high-level commands to help you adjust your strategy if needed. "
        "Carefully read the human command provided below. "
        "Decide if this command requires you to change your current heuristic strategy to better align with your teammate's actions. "
        "If a change is necessary, output a JSON object with a single key, 'ego_agent_new_heuristic', "
        "and set its value to one of the heuristic strategies listed above. "
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
