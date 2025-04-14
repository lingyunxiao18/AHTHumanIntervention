import os
import json
import openai

# Ensure your OpenAI API key is set in the environment variable 'OPENAI_API_KEY'
openai.api_key = os.getenv("OPENAI_API_KEY")

def process_command(user_command: str) -> dict:
    """
    Uses OpenAI's GPT-4 to process a human intervention command.
    Based on the input command, outputs a JSON object with a single key "ego_agent_new_heuristic".
    The value is expected to be either "clockwise", "counterclockwise", or null if no change is needed.
    
    Example output:
       {"ego_agent_new_heuristic": "counterclockwise"}
    """
    prompt = (
        "You are a strategic assistant that analyzes a human intervention command to generate "
        "instructions for a robotic cooking agent's heuristic adjustment in Overcooked. "
        "Based on the following human command, output a single JSON object with one key, "
        "'ego_agent_new_heuristic', whose value is either 'clockwise', 'counterclockwise', or null if no change is "
        "recommended. Do not include any extra commentary. Output valid JSON only.\n\n"
        f"Human command: \"{user_command}\""
    )
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful strategic assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
        )
        content = response.choices[0].message["content"]
        # Parse the JSON response
        result = json.loads(content)
    except Exception as e:
        print("Error during LLM processing:", e)
        result = {"ego_agent_new_heuristic": None}
    return result
